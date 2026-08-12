# OxiCUDA TODO

Pure Rust CUDA replacement for the COOLJAPAN ecosystem.
(C) 2026 COOLJAPAN OU (Team KitaSan)

## Project Status (v0.5.5 — 2026-08-13)

- **Crates**: 74 workspace members (73 library crates + 1 umbrella)
- **Files**: 4,521 Rust source files
- **Code**: ~1,312,651 SLoC (Rust)
- **Tests**: 38,987 passing (workspace-wide, `--all-features`), 37,629 with default features, 2 skipped (GPU-only on macOS), plus 272 doc tests passing workspace-wide (18 ignored, 74/74 crates — 17 of those in `oxicuda`, 4 in `oxicuda-metal`, 1 in `oxicuda-webgpu`) (all measured 2026-08-13); separately, 686 under `--features gpu-tests` on real Apple M3 hardware (measured 2026-08-12, not re-run this pass)
- **Warnings**: 0 (clippy + rustc, `-D warnings`)
- **unwrap() calls**: 0 (no-unwrap policy in library code)
- **New in 0.5.1**: `oxicuda-nvrtc` — pure-Rust runtime loader for NVIDIA's NVRTC (CUDA-C → PTX JIT) via `libloading` (zero SDK dependency, graceful degradation when absent); exposed from the umbrella behind the `nvrtc` feature. Brings the workspace to 74 members.
- **Status**: Vol.1–61 complete — Vol.1 Foundation, Vol.2 PTX/Autotune, Vol.3 BLAS, Vol.4 DNN, Vol.5 Scientific, Vol.6 Signal, Vol.7 Computation Graph, Vol.8 Training, Vol.9 Inference, Vol.10 RL, Vol.11 High-Perf Inference, Vol.12 Distributed, Vol.13 LLM Primitives, Vol.14–16 backend crates, **Vol.17 Generative AI**, **Vol.18 Graph Neural Networks**, **Vol.19 State Space Models (Mamba/S4/RWKV)**, **Vol.20 Vision Transformers & CLIP**, **Vol.21 Audio/Speech ML (Conformer/Wav2Vec2/CTC/WaveNet/SpecAugment/x-vector)**, **Vol.22 Time-Series Forecasting (TCN/NHiTS/PatchTST/TimesNet/iTransformer/RevIN)**, **Vol.23 Bayesian Deep Learning**, **Vol.24 Federated Learning**, **Vol.25 Neural Architecture Search**, **Vol.26 Self-Supervised Learning (SimCLR/MoCo/BYOL/Barlow/VICReg/MAE/SwAV/DINO)**, **Vol.27 Adversarial Robustness (FGSM/PGD/MIM/CW/AutoPGD/TRADES/MART/RS/IBP)**, **Vol.28 Multi-Modal Learning (cross-modal attn/CLIP/ImageBind/BERT/ViT/Conformer/DeepONet)**, **Vol.29 Continual Learning (EWC/SI/MAS/PackNet/Piggyback/ProgNN/GEM/DER++)**, **Vol.30 3D Geometry & Point Clouds (FPS/kNN/KD-tree/PointNet/PointNet++/DGCNN/ICP/Gaussian-splatting)**, **Vol.31 Physics-Informed Neural Networks (PINN/NeuralODE/FNO/DeepONet/adjoint-method)**, **Vol.32 RLHF & Alignment (DPO/IPO/KTO/ORPO/SimPO/reward-model/PPO-RLHF)**, **Vol.33 Meta-Learning (MAML/FOMAML/ANIL/Reptile/ProtoNet/MatchingNet/RelationNet)**, **Vol.34 Neural Radiance Fields (NeRF/Instant-NGP hash-grid/Mip-NeRF/TensoRF/volume-rendering)**, **Vol.35 Mixture of Experts (Switch/Top-K/Expert-Choice/Soft-MoE/SwiGLU/load-balance-loss)**, **Vol.36 Tabular Deep Learning (sparsemax/entmax15/TabNet/SAINT/FT-Transformer/NODE/QuantileNorm)**, **Vol.37 Anomaly Detection (DeepSVDD/AE/VAE/LOF/COPOD/Mahalanobis/IsolationScorer/MAD/ZScore/Ensemble)**, **Vol.38 Quantum Simulation (state-vector/gates/Pauli/VQE/QAOA/Trotter-Suzuki/density-matrix/Kraus-channels/QML-kernels)**, **Vol.39 Approximate Nearest Neighbor & Vector Search (HNSW/IVF/PQ/IVFPQ/LSH/MinHash/SimHash/NN-Descent/SQ4/SQ8)**, **Vol.40 Recommender Systems (ALS/BPR/NMF/NCF/TwoTower/DeepFM/AutoInt/WideDeep/GRU4Rec/SASRec/BERT4Rec/LightGCN/NGCF/MMoE/PLE/ESMM/neg-sampling/ranking-metrics)**, **Vol.41 Causal Inference (NOTEARS/PC/GES/NOTEARS-MLP/IPW/S-T-X-learners/AIPW/DML/DragonNet/2SLS/DeepIV/CausalForest/TwinNetwork/do-calculus)**, **Vol.42 Parameter-Efficient Fine-Tuning (LoRA/QLoRA/AdaLoRA/DoRA/IA³/Prefix-Tuning/P-Tuning-v2/Prompt-Tuning/Houlsby/Pfeiffer/Parallel/Compacter/BitFit/Diff-Pruning/TIES/DARE)**, **Vol.43 Knowledge Distillation (Hinton/DIST/DKD/FitNets/AT/PKT/RKD/CRD/CC/DML/BYOT/EMA/BAN/TAS/Progressive/DAFL/ZSKD)**, **Vol.44 Optimal Transport (Sinkhorn/Sinkhorn-divergence/network-simplex/EMD/W1/W2/Sliced/Max-Sliced/Gromov-Wasserstein/Fused-GW/Unbalanced-OT/JKO/Schrödinger-Bridge/Wasserstein-barycenter/multi-marginal/Wasserstein-kmeans/OT-domain-adaptation)**, **Vol.45 Spiking Neural Networks (LIF/IF/Izhikevich/AdEx/Poisson/sigmoid·atan·triangle·super-spike·fast-sigmoid surrogates/BPTT/STBP/SLAYER/STDP/R-STDP/triplet-STDP/ANN→SNN/threshold-balance/rate·TTFS·phase·Poisson encodings/spiking-linear·conv·pool·recurrent layers/Liquid-State-Machine/van-Rossum·Victor-Purpura·sync metrics)**, **Vol.46 Differential Privacy (exponential/Report-Noisy-Max/PTR mechanisms/SVT/AboveThreshold/f-DP·GDP·zCDP·tCDP·PRV accounting/strong·heterogeneous composition/Poisson·uniform·shuffling amplification/DP-FTRL·DP-Adam optimizers/GRR·OUE·RAPPOR local DP/local·smooth sensitivity/budget tracking)**, **Vol.47 Hyperdimensional Computing (binary·integer·complex hypervectors/XOR·circular-conv binding/majority-vote·superposition bundling/cyclic·random permutation/item·associative memory/HD classifier·online-update/record·n-gram·pattern encoding/Hamming·cosine·Jaccard distance/capacity bounds)**, **Vol.48 Evolutionary & Genetic Algorithms (CMA-ES/NSGA-II/MOEA-D/NEAT/DE/jDE/PSO/ACO/hypervolume/IGD)**, **Vol.49 Topological Data Analysis (Vietoris-Rips/persistent-homology/barcode/bottleneck-Wasserstein/Mapper/witness-complex/Betti-numbers/persistence-entropy/landscape)**, **Vol.50 Tensor Networks (MPS/MPO/two-site-DMRG/Lanczos/TEBD/Suzuki-Trotter-1st·2nd·4th/PEPS/boundary-MPS/TT-SVD/TT-cross/HOSVD/HOOI/CP-ALS/Jacobi-SVD/einsum/greedy-contraction-path/entanglement-entropy)**, **Vol.51 Sequence Models & Structured Prediction (HMM-discrete·Gaussian/forward-backward/Viterbi/Baum-Welch/linear-chain-CRF/L-BFGS/MEMM/structured-SVM/beam-search/Needleman-Wunsch/Smith-Waterman/Gotoh/Hirschberg/grid-CRF-mean-field/Kalman/RTS/EKF/EM/Ising-Gibbs/loopy-BP/edit-distance/BLEU)**, **Vol.52 Numerical PDE Solvers (FDM-Poisson-1D·2D/heat/wave/advection·upwind·Lax-Wendroff/FEM-P1-triangle-Dirichlet/Chebyshev-collocation/FFT-spectral/forward·backward·Crank-Nicolson·RK4·BDF2·IMEX/multigrid-V-cycle/CG·PCG-Jacobi·SSOR·ILU0/DG1D-LGL-Lax-Friedrichs)**, **Vol.53 Manifold Learning & Riemannian Geometry (PCA/Kernel-PCA/FastICA/t-SNE-Barnes-Hut/UMAP-fuzzy-simplicial/LLE·MLLE/Isomap-Dijkstra/Laplacian-Eigenmaps/Diffusion-Maps/Classical·SMACOF-MDS/KD-tree·Ball-tree-kNN/Jacobi-eig·Lanczos·Householder-QR/Stiefel·Grassmann·SPD·Poincaré-ball/Riemannian-SGD/trustworthiness·continuity)**, **Vol.54 Statistical Inference & Hypothesis Testing (erf·lgamma·digamma·betainc·gammp/Normal·Student-t·χ²·F·Beta·Gamma·Binomial·Poisson·Exponential PDF·CDF·PPF/one-sample·two-sample·Welch·paired-t/one-way·two-way-ANOVA/MANOVA-Wilks-Pillai/regression-SE-p-R²-F/Mann-Whitney·Wilcoxon·Kruskal-Wallis·Friedman/KS·Anderson-Darling·Shapiro-Wilk·Jarque-Bera/χ²-independence·Fisher-exact·McNemar/Bonferroni·Holm·BH·BY·Tukey-HSD/Bootstrap·Jackknife·Permutation/Wilson·Clopper-Pearson·Agresti-Coull-proportion-CI/Pearson·Spearman·Kendall-τ/OLS·Ridge·Logistic-IRLS/t·ANOVA-power·η²·partial-η²·ω²)**, **Vol.55 Streaming Data Sketches (Murmur3·FNV·xxH3·2-universal·tabulation hashes/HyperLogLog·HLL++·LinearCounting/Count-Min·Count-Sketch·conservative-update/Bloom·Counting-Bloom·Cuckoo/t-Digest·KLL·Greenwald-Khanna·P²-quantile/Misra-Gries·Space-Saving·Frequent-heavy-hitters/MinHash·SimHash·Weighted-MinHash/Cosine-LSH·Jaccard-LSH·banded-index/Reservoir·Weighted-Reservoir·Bernoulli·Priority-sampling/AMS-L2·Johnson-Lindenstrauss·Lp-stable-projection/Welford-online·exponential-decay·sliding-window)**, **Vol.56 Survival Analysis (Kaplan-Meier·Greenwood-SE·log-log-CIs/Nelson-Aalen-cumulative-hazard/log-rank·stratified·Peto-Peto·Gehan-Breslow/Cox-PH-Breslow·Efron-Newton-Raphson·Schoenfeld·Breslow-baseline/AFT-Exponential·Weibull·log-normal·log-logistic·generalised-gamma/time-varying-Cox·counting-process/Fine-Gray·cumulative-incidence·cause-specific/RMST-delta-method/Harrell·Uno-C/IPCW-Brier·integrated-Brier·time-dependent-AUC/deep-surv-head·partial-likelihood-grad·cox-loss)**, **Vol.57 Convex Optimisation (revised-simplex-LP·primal-dual-Mehrotra-IP/active-set·primal-dual-QP/SOCP·SDP-interior-point·log-det-barrier/ADMM·consensus-ADMM·augmented-Lagrangian/proximal-gradient·FISTA·Douglas-Rachford·Chambolle-Pock-primal-dual/L1·L2·L∞·group-lasso·elastic-net·nuclear·1D-TV-Condat·indicator-prox/simplex·L1·L2·box·PSD·SOC·halfspace-projection/projected-GD·Nesterov·Polyak-heavy-ball/Armijo·Wolfe·strong-Wolfe-line-search)**, **Vol.58 Compressed Sensing & Sparse Recovery (OMP·StOMP·ROMP·CoSaMP·SP-greedy/IHT·NIHT·HTP·AIHT-thresholding/AMP·VAMP·EB-AMP/Basis-Pursuit-LP·BPDN·Dantzig-Selector/LASSO-coord-descent·LARS·FISTA·group·fused·elastic-net/Chambolle-1D·2D-TV-denoise/SVT-matrix-completion·Nuclear-norm·ADMM/Robust-PCA-PCP·GoDec/Sparse-PCA-Witten/SBL-RVM·fast-marginal/K-SVD·MOD·online-DL/Gaussian·Bernoulli·partial-Fourier-measurement·RIP-estimator)**, **Vol.59 Classical Graph Algorithms (BFS·DFS·IDDFS·bidirectional-BFS/Kahn·DFS-topological/Dijkstra·Bellman-Ford·SPFA·Floyd-Warshall·Johnson·A*·Yen-k·bi-Dijkstra/Prim·Kruskal·Borůvka-MST·Union-Find/Edmonds-Karp·Dinic·Push-Relabel·min-cut/Hopcroft-Karp·Hungarian-Munkres·blossom-matching/Tarjan·Kosaraju·Gabow-SCC·bridges·articulation·biconnected/Brandes-betweenness·closeness·eigenvector·PageRank·Katz-centrality/Louvain·label-propagation·Girvan-Newman/Chu-Liu-Edmonds-arborescence/VF2-isomorphism/greedy·DSATUR·Welsh-Powell-coloring/Christofides·NN·2-opt·Held-Karp-TSP/Hierholzer-Eulerian)**, **Vol.60 Numerical Analysis Primitives (bisection·Newton·secant·Brent·Halley·Aberth-Ehrlich-root/Romberg·Gauss-Legendre·Hermite·Laguerre·Chebyshev·Clenshaw-Curtis·adaptive-Simpson·Gauss-Kronrod-G7K15-quadrature/Bessel-JYIK·Airy-Ai·Bi·Lambert-W·hypergeometric-2F1·elliptic-K·E·Riemann-ζ·dilogarithm·exponential-integral·digamma·trigamma/Euler·Heun·RK4·Dormand-Prince-DOPRI5·BDF1·BDF2·Rosenbrock-W·IMEX-Euler-ODE/Durand-Kerner·Jenkins-Traub·companion-QR-poly-roots/central-difference·Richardson·complex-step-diff/linear·natural-clamped-cubic-spline·Akima·PCHIP·Lagrange·Hermite·barycentric-interp/Monte-Carlo·Sobol·Halton-QMC·tensor-Gauss·Genz-Malik-cubature)**, **Vol.61 2D Computational Geometry (Point·Vector·Line·Segment·Ray·Circle·AABB·Polygon-primitives/orientation·in-circle·robust-predicates/segment-segment·line-line·circle-segment·circle-circle-intersection/winding·ray-cast·convex-O-log-n·in-circle-containment/Graham·Andrew·QuickHull·Jarvis·Chan-convex-hull/ear-clipping·Bowyer-Watson-Delaunay·constrained-Delaunay/Fortune-sweepline·Voronoi-from-Delaunay/Sutherland-Hodgman·Weiler-Atherton·Cohen-Sutherland·Liang-Barsky-clipping/shoelace-area·centroid·perimeter·convexity·offset·Minkowski-sum/divide-conquer-closest-pair/Welzl-smallest-circle·rotating-calipers·AABB/Bentley-Ottmann-sweepline/slab-trapezoidal-map-point-location/2D-KD-tree·R-tree-STR·quadtree)**; SDE samplers added to oxicuda-rand. **Wave AAA+51 (2026-06-08):** Quantum Fourier Transform + inverse + Quantum Phase Estimation (Vol.38 oxicuda-quantum `fourier/`); neuronal-avalanche criticality (branching σ, power-law MLE) + spike-train mutual-information/entropy (Miller-Madow) + population-vector & spike-triggered-average/covariance decoding (Vol.45 oxicuda-snn `metrics/`); Betti curves + persistence scale-space/PWGK/sliced-Wasserstein kernels + inclusion zigzag persistence with born-by-deletion (Vol.49 oxicuda-tda `vector/`, `distance/kernel`, `homology/zigzag`). **Wave AAA+52 (2026-06-08):** Parks-McClellan/Remez equiripple FIR design + Welch/Bartlett/sine-multitaper PSD estimation + polyphase rational resampling (Vol.6 oxicuda-signal `filter/remez`, `spectral/welch`, `resample/polyphase`); Hessian LLE + LTSA + non-metric MDS with PAVA isotonic regression (Vol.53 oxicuda-manifold `local/hessian_lle`, `local/ltsa`, `mds/nonmetric_mds`); k-core decomposition (Batagelj-Zaversnik O(V+E)) + boolean transitive closure + DAG transitive reduction (Vol.59 oxicuda-graphalg `connectivity/k_core`, `shortest_path/transitive_closure`·`transitive_reduction`). **Wave AAA+53 (2026-06-08):** 3D Delaunay tetrahedralization (Bowyer-Watson, empty-circumsphere) + Möller-Trumbore ray-triangle & Ericson point-triangle distance + discrete curvature (cotangent-Laplacian mean, angle-defect Gaussian, Gauss-Bonnet) (Vol.30 oxicuda-geometry3d `mesh/delaunay3d`·`ray_triangle`·`curvature`); IC(k) incomplete Cholesky + LOBPCG block eigensolver + smoothed-aggregation algebraic multigrid V-cycle (oxicuda-sparse `preconditioner/ick`·`amg`, `eig/lobpcg`, host-CSR CPU path); Cox martingale/deviance/cumulative-sum residuals + DFBeta influence diagnostics + Aalen-Johansen variance/confidence-bands (Vol.56 oxicuda-survival `cox/residuals_diagnostic`·`influence_diagnostics`, `nonparametric/multi_state_inference`) **Wave AAA+54 (2026-06-08):** Greiner-Hormann non-convex polygon Boolean (union/intersection/difference/xor, area-identity oracle) + 2D alpha shapes (Delaunay-dual circumradius filtration, convex-hull limit) + half-plane intersection (sorted-deque, empty/unbounded detection) (Vol.61 oxicuda-geom2d `clipping/greiner_hormann`, `alpha_shape/alpha_shape`, `halfplane/half_plane_intersection`); tensor-product Chebyshev-2D collocation Poisson (spectral accuracy) + Raviart-Thomas RT0 mixed FEM (per-element local conservation) + 2D nodal-P1 Discontinuous Galerkin with Cockburn-Shu/Zhang-Shu slope limiter (advection + inviscid-Burgers Rankine-Hugoniot) (Vol.52 oxicuda-pde `spectral/chebyshev_2d`, `fem/mixed_poisson`, `dg/dg_2d`·`limiter_2d`); KLL quantile-sketch merge (Karnin-Lang-Liberty, replay-into-levels) + weighted Misra-Gries heavy-hitters (Berinde-Cormode-Indyk-Strauss) + tug-of-war/AMS second-moment F2 with 4-wise-independent sign hashing (Vol.55 oxicuda-sketch `quantile/kll` merge, `topk/weighted_misra_gries`, `moment/ams_f2`+`hash/fourwise`) **Wave AAA+55 (2026-06-13):** Dykstra 1983 POCS (projection onto convex-set intersections with incremental Dykstra corrections, convergence to nearest point in ∩Cᵢ) + Boyd §7.2 Dual Decomposition (dual ascent for separable min Σfᵢ(xᵢ) s.t. ΣAᵢxᵢ=b, configurable step/max_iter/tol) + Mehrotra 1992 Predictor-Corrector QP (min ½xᵀPx+qᵀx s.t. Ax=b,x≥0 via affine predictor σ=0, centering parameter σ=(μ_aff/μ)³, corrector with cross-term dx_a⊙dz_a, 0.99 fraction-to-boundary) (Vol.57 oxicuda-cvx `projection/dykstra_pocs`, `admm/dual_decomp`, `qp/mehrotra_qp`); 387→418 tests (+31: 9+9+9 unit + 4 e2e), zero clippy warnings; ABOD (Kriegel-Schubert-Zimek KDD 2008, ABOF=Var[⟨pa,pb⟩/(‖pa‖·‖pb‖)²] over all training pairs, score=-ABOF: low variance=outlier; high=anomalous) + FastMCD (Rousseeuw-Van Driessen JASA 1999, C-step Minimum Covariance Determinant with n_starts random h-subset starts, Gauss-Jordan inversion with ridge 1e-5, Cholesky log-det convergence |Δ|<1e-10 relative) + COF (Tang-Chen-Fu 2002, Connectivity-based Outlier Factor: SBN greedy NN chain cost=(2/(k(k+1)))·Σᵢ i·dist(oᵢ₋₁,oᵢ), COF(o)=cost(o)/mean{cost(p): p∈kNN(o)}) (Vol.37 oxicuda-anomaly `distance/abod`, `density/fast_mcd`, `distance/cof`); 408→457 tests (+49: 14+16+17 unit + 2 e2e), zero clippy warnings. workspace 24,766→24,846 (+80 total) **Wave AAA+56–AAA+64 (through 2026-06-16, v0.2.0 release):** Extended Persistence & Discrete Morse theory (oxicuda-tda), Parametric UMAP (oxicuda-manifold), Fisher Information estimation (oxicuda-bayes), adaptive RK45 + Richardson extrapolation for ODE/PDE solvers, plus expanded CUDA kernel coverage across driver/memory/launch/backend layers; workspace 24,846→32,320 (+7,474 total). **Waves AAA+56–62 closed out (2026-06-20, v0.2.0 roadmap COMPLETE):** the 9 remaining roadmap algorithms were implemented (33 others already shipped under sibling filenames) — NODE/Neural-Oblivious-Decision-Ensembles with entmax routing (oxicuda-tabular `tree/node_oblivious`), Class-wise ECE + multiclass Brier decomposition (oxicuda-bayes `calibration/ece_classwise`), Gauss-Patterson nested quadrature via Golub-Welsch/Legendre + Smolyak sparse grid (oxicuda-numeric `quadrature/gauss_patterson` + new `linalg/tridiag_eig` QL eigensolver) and dual-number forward-mode autodiff (oxicuda-numeric `diff/automatic_diff`), SMC sequential compressed sensing / RVM particle filter with Sherman-Morrison updates (oxicuda-cs `sbl/smc_cs`), incremental Online-LOF with verified incremental≡batch equivalence (oxicuda-anomaly `distance/lof_online`), DP-HyperLogLog with 3 sensitivity-analysed mechanisms (oxicuda-sketch `cardinality/dp_hll`) + Spielman-Srivastava spectral graph sparsifier (oxicuda-sketch `matrix/graph_sketch`), ClusterMap unified attraction/repulsion embedding (oxicuda-manifold `reduction/clustermap`); plus the 4 stub fixes — kernel-fusion roofline cost-model (oxicuda-ptx), Cox-GB Newton leaf value (oxicuda-survival, also fixed a real NaN-denominator bug), ES-HyperNEAT quadtree node discovery (oxicuda-evol), and a faithful CPU model of CUDA stream-ordered allocation + memory-pool reuse (oxicuda-driver). workspace 32,320→32,426; zero clippy warnings; full nextest green. **Per-crate clean-CPU sweep (2026-06-20, same session, +185 tests → 32,611):** after closing the curated root roadmap, swept the per-crate TODOs and implemented the genuinely-missing CPU-testable algorithms (most named per-crate targets already existed under sibling filenames — stale checkboxes) — Sophia second-order optimizer (oxicuda-train, also added the crate's missing `LcgRng`), Niederreiter low-discrepancy sequence (oxicuda-rand), AWQ activation-aware weight quantization (oxicuda-infer), OptNet differentiable-QP via transposed-KKT (oxicuda-cvx `differentiable/kkt_diff`), NEAT + Novelty Search (oxicuda-evol), Swendsen-Wang + Wolff cluster MCMC (oxicuda-seq), MANOVA descriptive-discriminant follow-up (oxicuda-stats), Riemannian fixed-rank-manifold optimization (oxicuda-tn `optim/riemannian_tn`), Marginal-Structural Cox via IPTW (oxicuda-survival `cox/causal_cox`), neural linear-chain CRF + Pointer Network + Translation-Edit-Rate (oxicuda-seq), and complex Newton/Halley root-finding + deflation (oxicuda-numeric `root/complex_newton`). 4 latent bugs found & fixed with regression tests (Cox-GB NaN denominator, kkt_diff RHS sign, complex-Newton false-convergence-at-stationary-point, AWQ β-overshoot). SATURATION CONFIRMED: a final 10-candidate probe found 8 already-implemented (consistency-models, expert-choice, mixtral, saddle-point, multigrid-PCG, mamba2/SSD, S5, MACER all present). The remaining bulk-uncompleted per-crate items are (a) hardware-gated (real NVIDIA A100/H100 + driver 525+, Tensor-Core/TMA/FP8/cp.async PTX, ROCm/Vulkan/Metal/Level-Zero/WebGPU backends, on-device benches), (b) heavy transformer/vision/audio architectures already present, or (c) non-deterministic LLM-pipeline methods (e.g. Constitutional AI) — none cleanly CPU-testable here; further progress needs explicit user direction to scope the GPU/hardware class. **On-device GPU validation pass (v0.4.0, 2026-07-01):** for the first time, hand-written PTX kernels across 60+ crates were JIT-compiled (`Module::from_ptx`) via a new feature-gated `gpu-tests` harness and executed on a real NVIDIA RTX A4000 (sm_86) rather than only checked against a CPU-logic oracle, with every fix verified fail→revert→pass on the actual device. This found and fixed dozens of genuine defects: register-shadowing of built-in special registers (`%tid`/`%ntid`/`%ctaid`/`%warpid` clobbered by identically-named `.reg` declarations) across `oxicuda-primitives`, `oxicuda-train`, `oxicuda-ann`, `oxicuda-rl`, `oxicuda-dist-infer`, `oxicuda-timeseries`; base-2-vs-base-e log/exp mixups (`ex2.approx`/`lg2.approx` used where `ln`/`log` was needed) in `oxicuda-survival`, `oxicuda-seq`, `oxicuda-ot`, `oxicuda-rlhf`, `oxicuda-nerf`, `oxicuda-gnn`, `oxicuda-audio`; invalid PTX rejected outright by `ptxas` — bad/duplicate registers, illegal shared-memory addressing, malformed branch labels, missing predicate declarations — in `oxicuda-multimodal`, `oxicuda-quant`, `oxicuda-moe`, `oxicuda-geom2d`, `oxicuda-distill`, `oxicuda-recsys`, `oxicuda-ann`, `oxicuda-infer`, `oxicuda-causal`, `oxicuda-lm`, `oxicuda-tda`, `oxicuda-fft`, `oxicuda-evol`, `oxicuda-cs`, `oxicuda-sparse`, `oxicuda-privacy`; and kernels that were still bare `ret;` stubs behind a real, already-tested CPU reference — most notably `oxicuda-solver`'s LU/Cholesky panel-factorization kernels, now real `bar.sync`-staged implementations. Alongside the sweep, several partial/proxy PTX kernels were completed to the full algorithm: `fem_assemble_kernel` (`oxicuda-pde`, full P1 stiffness assembly validated against `p1_local_stiffness`), `sparsemax_kernel`/`quantile_norm_kernel`/`node_tree_eval_kernel` (`oxicuda-tabular`, exact Martins-Astudillo sparsemax + empirical-CDF quantile normalization + full multi-level NODE tree replacing a 2-leaf hardcode), `soft_moe_dispatch_kernel` (`oxicuda-moe`, real 3-pass slot-softmax dispatch), `project_kernel`/`sh_eval_kernel` (`oxicuda-geometry3d`, full EWA covariance + all 9 L=0..2 spherical-harmonic terms), and 4 previously-empty-loop `oxicuda-recsys` kernels (`embedding_lookup`/`dot_score`/`bpr_gradient`/`lightgcn_propagate`). Workspace tests grew 32,611→38,093 (+5,482), zero clippy warnings, zero unwrap() in library code. **PTX F64 codegen + GEMM precision fixes (v0.5.0, 2026-07-14):** the `oxicuda-ptx` elementwise/reduction kernel templates previously generated invalid PTX at F64 precision — wrong-width value registers (fixed via a precision-aware register prefix) and an F32-only warp-shuffle reduction tail (F64 now reduces via shared-memory tree) — both now `ptxas`-clean; F32-only special-function-unit ops (exp/log/sigmoid/gelu/silu/tanh/softplus/pow) now reject non-F32 precision up front instead of emitting invalid PTX. `oxicuda-blas`'s GEMM `alpha`/`beta` scalars were encoded at the wrong (input, not accumulator) precision for F16/BF16/FP8 inputs, silently corrupting mixed-precision GEMM results; fixed via a new `GpuFloat::to_accumulator_bits()` conversion. Workspace tests grew 38,093→38,622 (net, since 0.4.0; 38,612→38,622 since 0.4.1), and the last 42 production `.expect()` call sites across 22 crates were eliminated, completing the workspace's zero-unwrap policy.

## GEMM skinny/small-M occupancy + cross-stream readback race (2026-08-12) [VERIFIED + FIXED]

**Context:** downstream investigation into why `oxiface` (Rust face-swap CLI) underperforms on Linux+NVIDIA vs. its CoreML path traced a large share of it to `oxicuda-blas` GEMM for the exact shapes ArcFace/InSwapper dominate inference with (`M=1` or `M=8`, `K=25088`, `N=512` — a `1x512` embedding/emap projection). Both findings below were verified on-device (RTX A4000, sm_86, driver 550.144.03) with fail-before/pass-after regression tests, not just read from source.

1. **`Split-K parallelization (splitk.rs)` was marked `[x]` complete above (BLAS Level 3 checklist) but was never wired into `GemmDispatcher::dispatch`.** `SplitKConfig` and `generate_splitk_reduction_kernel` existed, were unit-tested in isolation, and `dispatch.rs::splitk_tile_config` even computed a `split_k` factor for the `SplitK` category — but `dispatch()` only ever compiled/launched `GemmTemplate::generate()`'s single-pass kernel, which does one thread per output element with a *full, serial* K-length reduction inside that thread and folds any `gridDim.z` into extra (redundant, for skinny shapes already-saturated) coverage of the M*N space, never a K-partition. For `Skinny`-category problems (`M<32` or `N<32`, the ArcFace/InSwapper shape) this caps the launch at `M*N` total threads *no matter how large K is* — for `M=1, N=512` that's exactly 512 threads on a device that can schedule ~70k concurrently, each doing a 25088-element serial reduction. Fixed by adding a real second kernel (`splitk::generate_splitk_partial_kernel` — `gridDim.z` threads each reduce a disjoint K sub-range into a scratch workspace) and wiring `GemmDispatcher::dispatch_skinny_split_k` to route `Skinny`+small-`M*N`+large-`K` NoTrans×NoTrans F32/F64 problems through partial-kernel-then-`generate_splitk_reduction_kernel` instead of the single-pass launch. Measured **~17.6x** speedup on the exact ArcFace shape (2525µs → 144µs/call, `oxicuda-blas` release build, warm kernel cache). New coverage: `oxicuda-blas/tests/gemm_shape_sweep_gpu.rs` (M∈{1,2,7,8,16,17,32,64,100,128,1024,2048} × several K/N, plus explicit 1024³/2048³ — every element checked against an `f64` CPU oracle, not sampled; a sibling stream had measured 1024³/2048³'s *throughput* only, never its numeric correctness — now covered too, and correct) and new unit tests in `level3/gemm/splitk.rs`.

   *Investigation hypothesis vs. what actually reproduced:* the report hypothesised the single-pass kernel's grid/block computation left output elements uncovered (reads back `0.0`, i.e. an untouched fresh allocation). On-device reproduction of that exact claim (`compute_grid`/`compute_block`/`skinny_tile_config` driving `GemmTemplate::generate()` directly, bypassing every other layer) found **zero wrong elements** across the same M-sweep — `generate()`'s kernel derives its grid-stride bound from the launch's own runtime `gridDim`/`blockDim`, which is correct for any grid, including the exact undersized one measured. The occupancy problem was real and worth fixing (above); the *numeric*-wrongness claim traced to a second, independent bug:

2. **`oxicuda-dnn::DnnHandle` gives its internal `BlasHandle` its own CUDA stream** (`build()`: "Create a separate stream for the internal BLAS handle so that BLAS and DNN launches can be overlapped when appropriate") — correct and deliberate, but two `oxionnx-cuda` call sites (`matmul::cuda_matmul`, `reduce.rs`'s `ReduceSum`/`ReduceMax`) dispatched through `ctx.dnn.blas()` and then synchronized `ctx.dnn.stream()` — the *other*, empty stream — before reading the result back to the host. Every `oxicuda-driver` stream is `CU_STREAM_NON_BLOCKING`, so the two streams do not implicitly order against each other: the host could read `C` back before the BLAS-stream kernel had actually finished, observing whatever the freshly-allocated/zeroed buffer already held. On real hardware this reproduced exactly as "numerically wrong GEMM" — `M=1,K=25088,N=512`: 439/512 elements wrong; `M=8` (same shape): 3220/4096 wrong; `64x64x64` all-ones: 3456/4096 wrong (expected 64.0 uniformly) — and the *inconsistent* wrong-element cutoff between the two `K=25088` shapes (both dispatch identically) is the signature of a race, not a coverage bug (a coverage bug cuts off at a consistent, shape-derived boundary). Fixed with a new `DnnHandle::synchronize_all()` (waits for both streams; documented with the full rationale so the next caller doesn't reintroduce this) and routing both call sites through it. `reduce.rs` carried a comment asserting `reduce_axis` "launches on `handle.stream()` — the same stream `ctx.dnn` owns" — false (`reduce_axis`'s `handle` parameter *is* the BLAS sub-handle, with its own stream); comment corrected alongside the fix. New coverage: `oxicuda-dnn`'s `gpu_tests/handle_sync.rs` (BLAS-dispatched GEMM read back after `synchronize_all`, correct) and `oxionnx-cuda/tests/matmul_shape_sweep_gpu.rs` (the exact repro shapes above, element-for-element against `reference::ref_matmul`, through `matmul::cuda_matmul` end to end — this is the test that would have caught the bug originally: fails against the pre-fix code with the exact wrong-element counts quoted above, passes after).

Both fixes are dev-only against the local `oxicuda`/`oxionnx` checkouts wired via `[patch.crates-io]` path patches in `oxionnx`/`oxiface`'s `Cargo.toml` — not yet published.

## 🚨 Production Readiness Audit — Wave PR-1 (2026-07-06) [VERIFIED + FIXED]

**Goal:** harden OxiCUDA for production (many downstream projects will depend on it). This is a multi-agent adversarial audit for latent bugs, security, memory-safety, resource leaks, and performance-degradation seeds — NOT feature work.

### Resolution status (2026-07-06 — verify + fix complete)
- **Verified:** all 109 findings were adversarially re-verified on the real RTX A4000 (sm_86) by a 124-agent workflow (one read-only verifier per file, default-reject, + a refute pass over every confirmed critical/high). **106 of 109 were confirmed real** (96 CONFIRMED + 10 PARTIAL); only **3 were false positives** — a 97% true-positive rate.
- **Fixed:** **95 findings fixed** by scope-locked per-crate agents (Opus for oxicuda-ptx/blas/driver, Sonnet elsewhere). Every touched crate is build-clean, clippy-clean (zero warnings), and its `--all-features` test suite (incl. on-device `gpu-tests`) passes. **Full-workspace gate is green: `cargo build --workspace` + `cargo build --workspace --all-features` + `cargo clippy --workspace --all-targets --all-features` (0 warnings), and a 10,488-test on-device `--no-fail-fast` run across every changed crate = 0 failures.** PTX-ISA fixes (WMMA/MMA/WGMMA/tex/mbarrier/`.maxntid`/64-bit GEMM) were empirically assembled by the box's real `ptxas`; the GEMM transpose fix (F001) is covered by a new on-device CPU-oracle regression test.
- **Two additional pre-existing bugs found + fixed while validating the fixes on-device (were latent, masked by the very defects this wave fixed):** (1) f32 SYRK/SYR2K used a broken `syrk_tc`/`syr2k_tc` placeholder (k=0 term only) and the GEMM fallback overwrote the off-triangle → replaced with a triangle-masked GEMM in `oxicuda-blas`; (2) the umbrella `oxicuda` backend fed **ColMajor** descriptors to `oxicuda_blas::gemm` (which now correctly rejects them per F017) → rewired it to map column-major onto tight row-major via the transpose identity, so `identity*B == B` now holds end-to-end on-device.
- **On-device `compute-sanitizer` baseline:** memcheck over oxicuda-memory (257) / driver (439) / launch (214) device tests → **zero memory-safety violations** (all reported "errors" are expected negative-path CUDA API error codes). NOTE: the memory-safety findings below are *latent unsafe-API* holes (a safe `fn` exposing an `unsafe` contract) — they are proven by signature/code review, not by sanitizing correct-usage tests.
- **Deferred (8) — coordinated soundness/context batch:** see the dedicated section at the end. Eight are real soundness holes whose correct fix (`unsafe fn` / lifetime) ripples across the whole workspace (e.g. `Kernel::launch` = 218 call sites in 63 crates); doing a subset would leave an inconsistent API. A ninth item, **F025** (`Stream::new` context binding), had been implemented and passed the driver suite but was **reverted** because it surfaced a latent context-wiring bug in the umbrella `oxicuda` backend (a throwaway "token" context handed to `BlasHandle`); **it was fixed together with that backend on 2026-07-07** (see the F025 entry below) and no longer counts toward this deferred batch. Grouped for one deliberate breaking-change PR at the next version bump.
- **Skipped (2):** F058 (blas — the recommended fix routes through an unsound CPU-model stream-alloc; needs a real stream-ordered path), F102 (launch — `KernelArgs::as_param_ptrs` return-type change is api-breaking; belongs with the soundness batch).
- **Dropped (3 false positives):** F095 (autotune power_aware `nvidia-smi` PATH — threat model requires an attacker who already owns the process env), F105 (memory pool lock-poison handling — defensive hygiene, not a defect), F015 (blas dot workspace free — refuted).
- **Remaining GAP (partially closed 2026-08-12):** **alt-backends** (oxicuda-vulkan / metal / rocm / levelzero / webgpu) were queued for audit but their 5 agents hit the session token limit before running — **still un-audited, 0 findings collected.** A follow-up session audited and fixed **oxicuda-metal** and **oxicuda-webgpu** on real Apple M3 hardware (Metal 4, macOS). Headline finding: `oxicuda-metal`'s `conv2d_forward`/`attention` were false completions — silent CPU loops with finished MSL kernels never called — and `softmax` returned `Unsupported` despite a working shader; all three now dispatch real GPU kernels, and GEMM widened to all four transpose combinations. `oxicuda-webgpu` had no comparable false completion, but gained real device-limit probing (previously capped at the WebGPU conformance baseline), a non-fatal GPU-error handler, and checked-overflow batched-GEMM math. Full writeup: `CHANGELOG.md` `[0.5.5]` and the `oxicuda-metal`/`oxicuda-webgpu` entries below (~line 2493). **oxicuda-vulkan / oxicuda-rocm / oxicuda-levelzero remain entirely un-audited, 0 findings collected** — re-run next session.
- **Legend below:** `- [x]` fixed/verified · `- [ ] ⏸️DEFERRED` soundness batch · `- [ ] ⏭️SKIPPED` needs different fix · `- [ ] ❌DROPPED` false positive.

### Cross-cutting themes (the high-value clusters)
1. **Soundness holes — `unsafe` operations exposed as safe `fn`** (the biggest class): async H2D/D2H copies over borrowed host memory, `register_*`, unified/zero-copy host slices, `Kernel::launch`, `cooperative_launch`, arg-pointer builders. All UB reachable from 100% safe code. Fix pattern: make them `unsafe fn` with documented contracts, or add lifetime/stream-token guards.
2. **32-bit index/count arithmetic** in generated kernels + host grid math → silent wrong results (or zero work) on buffers ≥ 2³² elements / matrices ≥ 4 GiB — reachable on this 16 GB A4000 and routine on 40/80 GB GPUs. Fix: 64-bit offset math (`mul.wide.u32`) + host-side guards.
3. **"Stream-ordered" that ignores the stream** — the memory pool, `copy_dtod_async`, several BLAS workspaces free/recycle device memory without stream synchronization → aliasing / use-after-free / silent stale data.
4. **FFI discriminant & signature errors** — wrong `CUdevice_attribute` / `CUlaunchAttributeID` values, unversioned/`_v2` symbol mismatch, `cuMemcpyBatchAsync` fabricated 6-param signature (UB on CUDA 12.8+).
5. **Advertised-but-inert kernels/APIs** — `copy_3d_dtod`, `CooperativeLaunch`, `cluster_launch`, `GraphExec` (empty nodes), batched-GEMM arg mismatch, WMMA/MMA/WGMMA PTX that ptxas rejects, transposed GEMM ignoring `trans_a/b`. Return `Ok`/success while doing nothing or wrong.
6. **PTX cache security** — world-writable `/tmp` fallback with predictable names + symlink-following writes = local kernel-poisoning; non-atomic writes = torn files; cache key omits generator/ISA version.
7. **Per-call JIT & no module cache** in the BLAS crate (and likely others) — every op re-generates + `cuModuleLoadData`-compiles PTX = 10–100 ms tax per call. The single biggest perf-degradation seed.


### 🔴 CRITICAL (12)

- [x] **[correctness] oxicuda-blas** — `crates/oxicuda-blas/src/level3/gemm/dispatch.rs:485`
  - **Defect:** gemm() silently ignores trans_a/trans_b: the generated kernel always computes NoTrans A*B, so every transposed GEMM (and the SYRK/SYR2K fallbacks built on it) returns wrong results
  - **Impact:** gemm with Transpose::Trans/ConjTrans — the single most common BLAS variation — returns numerically wrong matrices with Ok(()) status; syrk/syr2k fallback results are wrong for f64 and small/Full-fill f32. Any downstream project doing C = A^T*B on this stack silently trains/solves on garbage.
  - **Fix:** Add trans_a/trans_b to GemmTemplate and emit the four indexing variants (or swap operand pointers plus dims for the row-major identity op(A)*op(B) = (op(B)^T*op(A)^T)^T), and add on-device gemm tests for all four transpose combinations.
- [x] **[correctness] oxicuda-blas** — `crates/oxicuda-blas/src/level2/gemv.rs:279` _(×2 auditors)_
  - **Defect:** Systemic 32-bit index/offset arithmetic in generated Level-1/2/3 BLAS and DNN kernels — GEMV wraps at a 4 GiB matrix, silently corrupting results (reachable on this machine's 16 GB A4000)
  - **Impact:** Silent numeric corruption on valid inputs: any GEMV over a matrix >= 4 GiB (fits even a 16 GB workstation GPU), and any level-1/2/3 or DNN kernel touching a buffer >= 2^32 elements (8 GiB f16 — routine on 40/80 GB datacenter GPUs, e.g. long-context attention with batch 8 x 32 heads x 128k seq x 128 dim = 4.3e9 elements) reads/writes wrapped in-bounds addresses. No error, no NaN — plausible-looking wrong numbers feeding training/inference.
  - **Fix:** Emit 64-bit offset math: replace the u32 stride/index multiplications with mul.wide.u32 (mul_wide_u32_to_u64) followed by u64 mad/add, as the Hopper flash-attention and cooperative-GEMM paths already do. Add a debug-assert or host-side check rejecting buffers whose required element index exceeds u32 range for kernels not yet converted.
- [x] **[memory-safety] oxicuda-blas** — `crates/oxicuda-blas/src/batched/strided_gemm.rs:243`
  - **Defect:** Entire batched-GEMM family launches an 8-parameter kernel with mismatched 5/13/17-field argument tuples — wild device-pointer loads AND stores corrupt arbitrary CUDA-context memory
  - **Impact:** Any call to the public batched/strided/grouped GEMM APIs makes the GPU dereference and WRITE through pointers assembled from unrelated scalar arguments — corrupting other tensors, other libraries' allocations, or faulting the whole CUDA context. This is silent when the garbage address happens to land inside a mapped allocation.
  - **Fix:** Implement real batched/strided kernels whose PTX parameter lists match the argument tuples (lda/ldb/ldc/ldd, strides, batch index via ctaid.z, pointer-array indirection, transpose handling), or route these APIs through per-batch calls to the existing dispatcher with the correct 8-argument tuple. Add an on-device test per public batched entry point.
- [ ] **[memory-safety] oxicuda-driver** — `crates/oxicuda-driver/src/cooperative_launch.rs:478` ⏸️DEFERRED (soundness batch)
  - **Defect:** cooperative_launch / cooperative_launch_multi_device are safe fns that make the driver dereference caller-supplied raw pointers (kernel args) — UB reachable from safe code
  - **Impact:** Any downstream crate can crash the process or corrupt host memory through the driver's argument-marshalling read without writing a single `unsafe` block; a dangling or forged arg pointer in a launch path becomes a heap-read of freed memory inside libcuda. This is an unsound public API in a crate positioned as the safe CUDA foundation for many projects.
  - **Fix:** Mark both launch functions (and any API accepting `*mut c_void` kernel params or raw `CUfunction`/`CUstream` handles) as `unsafe fn` with documented preconditions, or provide a typed-argument safe wrapper (e.g. an `impl KernelArg` trait over owned values) that keeps the argument storage alive across the FFI call.
- [x] **[memory-safety] oxicuda-driver** — `crates/oxicuda-driver/src/ffi_constants.rs:108`
  - **Defect:** CU_POINTER_ATTRIBUTE_IS_MANAGED defined as 7 (= BUFFER_ID) — driver writes 8 bytes into caller's 4-byte u32, corrupting the stack
  - **Impact:** Every call to HostRegisteredMemory pointer-info on a live CUDA context performs a 4-byte out-of-bounds stack write (silent memory corruption / UB that ASan or a shifted stack layout can turn into crashes or corrupted locals), and reports is_managed=true for ordinary unmanaged allocations, misleading any downstream unified-memory logic.
  - **Fix:** Set CU_POINTER_ATTRIBUTE_IS_MANAGED = 8 in ffi_constants.rs; fix CUpointer_attribute::IsManaged to 8 and DeviceOrdinal to 9 in ffi.rs; update the self-consistent tests. Audit the host_registered.rs call site to pass a buffer sized for the attribute's documented value type.
- [ ] **[memory-safety] oxicuda-launch** — `crates/oxicuda-launch/src/kernel.rs:246` _(×2 auditors)_ ⏸️DEFERRED (soundness batch)
  - **Defect:** Kernel::launch is a safe fn passing unchecked, arity-unverified argument pointers to cuLaunchKernel - host OOB reads and arbitrary device writes from safe code
  - **Impact:** A single wrong-arity or wrong-type launch (trivially written in safe code, e.g. after a kernel signature change) causes host-process UB via driver OOB pointer dereferences, or silent GPU memory corruption via type-confused pointer parameters - in the primary launch path of the whole stack.
  - **Fix:** Make Kernel::launch an `unsafe fn` (and the launch! macro emit an unsafe block with documented contract), or add a checked launch path that validates parameter count/sizes against cuFuncGetParamInfo / PTX .param metadata before calling cuLaunchKernel.
- [x] **[correctness] oxicuda-memory** — `crates/oxicuda-memory/src/copy_2d3d.rs:458`
  - **Defect:** copy_3d_dtod validates arguments then returns Ok(()) without copying anything (silent fabricated success)
  - **Impact:** Silent data corruption: callers doing volumetric copies get Ok(()) while the destination retains stale/uninitialized contents. Downstream numeric results are wrong with no error signal anywhere.
  - **Fix:** Return Err(CudaError::NotSupported) as documented until cuMemcpy3D_v2 is wired through the loader; then build the CUDA_MEMCPY3D descriptor and invoke it, mirroring copy_2d_dtod.
- [ ] **[memory-safety] oxicuda-memory** — `crates/oxicuda-memory/src/device_buffer.rs:297` _(×3 auditors)_ ⏸️DEFERRED (soundness batch)
  - **Defect:** Safe async copy APIs let host buffers be freed/reused while DMA is in flight (use-after-free from safe code)
  - **Impact:** Heap corruption / write-after-free in production: any downstream crate that drops or reallocates (Vec push/resize) a host buffer before stream synchronization gets the DMA engine scribbling over freed or reused allocator memory — nondeterministic crashes and silent data corruption that ASAN will attribute to unrelated code.
  - **Fix:** Make the raw-slice async variants `unsafe fn` with documented contracts, or return a guard token borrowing the slice until Stream::synchronize (lifetime-bound `PendingCopy<'a>` that syncs on drop). For PinnedBuffer/DeviceBuffer, record the last stream used and synchronize it in Drop before cuMemFreeHost/cuMemFree.
- [ ] **[memory-safety] oxicuda-memory** — `crates/oxicuda-memory/src/host_registered.rs:350` _(×2 auditors)_ ⏸️DEFERRED (soundness batch)
  - **Defect:** register_vec/register_slice/register return an unbound RegisteredMemory that aliases and outlives the source allocation (UAF + &mut aliasing from safe code)
  - **Impact:** Use-after-free and aliased mutation in any production consumer that registers a Vec for DMA and later grows/drops it while the handle is alive; also DMA into freed pages via the registered device_ptr. Miri/ASAN-visible UB; silent corruption otherwise.
  - **Fix:** Add a lifetime: `RegisteredMemory<'a, T>` holding `&'a mut [T]` (or make `register` an `unsafe fn` and delete the safe Vec/slice conveniences in favor of lifetime-bound wrappers). Remove Deref/DerefMut or derive them from the held borrow.
- [x] **[concurrency] oxicuda-memory** — `crates/oxicuda-memory/src/pool.rs:292` _(×4 auditors)_
  - **Defect:** PooledBuffer 'stream-ordered' pool ignores the stream: Drop recycles the device pointer immediately while enqueued GPU work may still use it, so alloc_async can hand the same pointer to a second concurrent user
  - **Impact:** Two GPU streams concurrently alias one device allocation → nondeterministic silent numeric corruption in production workloads that drop pooled buffers before synchronizing (the exact usage pattern the API's own docs advertise).
  - **Fix:** Record an event on the buffer's stream at drop and only move the pointer to free_bins once the event completes (query on alloc), or delegate to the real cuMemFreeAsync/cuMemAllocFromPoolAsync path (NativeMemoryPool) and delete the misleading software pool semantics.
- [x] **[correctness] oxicuda-ptx** — `crates/oxicuda-ptx/src/templates/gemm.rs:267`
  - **Defect:** Default GEMM kernel computes its grid-stride loop bound M*N in 32-bit — GEMMs with >= 2^32 output elements silently compute nothing (or a fraction) and return success
  - **Impact:** Production LLM shapes hit this exactly: a prefill logits GEMM with m = batch*seq = 16384 and n = vocab = 262144 gives M*N = 2^32, so the kernel computes ZERO elements and returns Ok — an 8 GiB f16 output of pure garbage with no error. Any C matrix > 4Gi elements (8 GiB f16 / 16 GiB f32, routine on A100-80G/H100) gets silently partial results. All downstream crates (oxicuda-dnn im2col conv uses this same gemm_api) inherit the corruption.
  - **Fix:** Compute total_elems and the loop index in u64 (mul.wide.u32 for M*N, 64-bit idx/compare), and use mad.wide.u32 for the A/B/C element offsets. Alternatively add a host-side guard in GemmDispatcher::dispatch rejecting problems where m*n, m*k, or k*n >= 2^32 with an explicit error until the kernel is 64-bit clean.
- [x] **[numeric] oxicuda-ptx** — `crates/oxicuda-ptx/src/builder/body_builder/math_f64.rs:163`
  - **Defect:** exp_f64/erf_f64/tanh_f64 silently return garbage (e.g. ~1e150 instead of 1.0) when the internal 2^k exponent assembly under/overflows — no clamping on a path used by production kernels
  - **Impact:** Silent numeric corruption in downstream GPU kernels: Sinkhorn optimal-transport iterations (oxicuda-ot) and log-normal sampling (oxicuda-rand) receive ~1e150-magnitude values or -0.0 where 0.0/1.0/inf are mathematically required, corrupting transport plans and distributions with no error signal.
  - **Fix:** Clamp k to [-1075, 1024] before the bias/shift and emit selp-based saturation: return 0.0 for k below the subnormal range and +inf (0x7FF0000000000000) on overflow; handle exponent 0x7FF (inf/NaN) explicitly in log_f64; add clamping in tanh_f64 (|x| > 20 → ±1).

### 🟠 HIGH (33)

- [x] **[correctness] oxicuda-autotune** — `crates/oxicuda-autotune/src/export.rs:461`
  - **Defect:** force_save() permanently corrupts the result DB by writing median_us = 0.0
  - **Impact:** Importing a bundle with the 'always' policy writes a permanently-unreplaceable, internally-inconsistent 0.0us record into the on-disk results DB; Dispatcher then serves that config and downstream throughput math divides by zero.
  - **Fix:** Do not mutate median_us. Give ResultDb an explicit force-insert method (e.g. `insert_overwrite`) that replaces unconditionally while preserving the real timing fields, and call that from force_save instead of poisoning median_us.
- [x] **[correctness] oxicuda-autotune** — `crates/oxicuda-autotune/src/parallel_bench.rs:466`
  - **Defect:** ParallelBenchmarkEngine::benchmark_parallel times an empty no-op closure, fabricating results
  - **Impact:** A production caller using benchmark_parallel gets plausible-looking but meaningless timings/GFLOPS and a meaningless 'best' config, silently corrupting any tuning decision or persisted DB built on top of it.
  - **Fix:** Change the API to accept a `Fn(&Config, stream_idx) -> Result<(), _>` launch closure and time that, or delete/feature-gate the engine and clearly mark it non-functional. Never return fabricated timings from a benchmarking API.
- [ ] **[memory-safety] oxicuda-blas** — `crates/oxicuda-blas/src/level1/dot.rs:88` ❌DROPPED (refuted)
  - **Defect:** dot/nrm2/asum/iamax free their per-call partials workspace immediately after async launches with no stream sync — in-flight use-after-free hazard the codebase itself documents elsewhere
  - **Impact:** On drivers/configurations where cuMemFree does not fully barrier, the phase-2 kernel reads freed (potentially reallocated) device memory — silent wrong scalar results; on drivers where it does barrier, every dot/nrm2/asum/iamax imposes an implicit device-wide sync, serializing all concurrent streams.
  - **Fix:** Synchronize handle.stream() before the workspace drops (matching trmm/softmax), or better: keep a reusable per-handle workspace / stream-ordered allocation so neither a free nor a sync happens per call.
- [x] **[memory-safety] oxicuda-blas** — `crates/oxicuda-blas/src/level2/gemv.rs:265`
  - **Defect:** MatrixDesc.layout is ignored by gemv/ger/symv/trmv/trsv and gemm: ColMajor descriptors produce device out-of-bounds reads (gemv, rows>cols) and silently wrong results everywhere
  - **Impact:** A perfectly valid ColMajor descriptor built through the validated constructor makes the GEMV kernel read thousands of elements past the buffer (potential context fault / garbage results); every other layout-blind op silently computes on transposed data. Padded sub-matrix views via with_ld silently void the constructor's size validation and are then ignored by gemm anyway.
  - **Fix:** Either honour layout/ld in every kernel (pass ld, branch on layout like trmm/trsm's elem_strides) or reject non-RowMajor/non-tight descriptors up-front with BlasError::InvalidArgument in gemm/gemv/ger/symv/trmv/trsv and re-validate buffer capacity when with_ld increases ld.
- [x] **[concurrency] oxicuda-blas** — `crates/oxicuda-blas/src/level2/trmv.rs:67`
  - **Defect:** trmv interleaves an unordered legacy-stream sync D2D copy with a non-blocking-stream kernel, and frees the copy while the kernel may still be reading it
  - **Impact:** Timing-dependent silent wrong results for trmv in any pipelined workload (x still being produced on the handle stream), and a latent use-after-free of x_copy that becomes live the moment module caching removes the accidental JIT delay between copy and launch.
  - **Fix:** Perform the snapshot with a stream-ordered copy on handle.stream() (cu_memcpy_dtod_async_v2), and synchronize the stream (or use stream-ordered free) before dropping x_copy — mirroring the trmm.rs pattern.
- [x] **[memory-safety] oxicuda-dnn** — `crates/oxicuda-dnn/src/conv/descriptor.rs:280`
  - **Defect:** conv_forward never checks the user's output tensor dims (or filter C/g) against the computed convolution shape — kernel writes batch*K*P*Q elements regardless, device OOB write
  - **Impact:** A wrong (but internally consistent) output-tensor shape argument silently corrupts unrelated device memory across the CUDA context; a wrong filter channel dim reads out of bounds. Reachable from the safe validated-constructor API surface used by every downstream framework.
  - **Fix:** In ConvProblem::validate (or from_descriptors), compare output.dims against [batch, out_channels, output_dims()...] and filter.dims[1] against in_channels/groups, returning DnnError::InvalidDimension on mismatch — mirroring pool/max_pool.rs:67.
- [x] **[numeric] oxicuda-dnn** — `crates/oxicuda-dnn/src/conv/fprop/im2col_gemm.rs:183`
  - **Defect:** DNN host-side element counts computed in u32 (or truncated from usize numel) — wraps in release builds, panics in debug, silently under-launching kernels
  - **Impact:** Large-batch convolutions overflow this realistically (batch 256, 224x224 output, C=64, 3x3 filter -> 7.4e9 > 2^32; ~15 GB f16 workspace, feasible on 40/80 GB GPUs): release builds produce silently wrong conv output, debug builds panic inside a library call. Resize/pool ops on >=4Gi-element tensors silently process a fraction (or nothing).
  - **Fix:** Compute all element counts in u64/usize (matching workspace_bytes), then u32::try_from with a DnnError::InvalidDimension on overflow before building kernel args; for numel-based ops validate output.numel() <= u32::MAX explicitly instead of casting.
- [x] **[correctness] oxicuda-driver** — `crates/oxicuda-driver/src/event.rs:44`
  - **Defect:** Module, Event, and GraphExec carry no context tether (unlike Stream) — safe code can destroy the Context first, then their Drops call cuEventDestroy/cuModuleUnload/cuGraphExecDestroy on stale handles
  - **Impact:** Warn-level log spam at best; at worst, handle-recycling turns a stale Drop into destruction of a live unrelated GPU resource in another context, producing hard-to-diagnose failures (kernels/events vanishing) in long-running multi-context services.
  - **Fix:** Mirror Stream's design: have Event::new/Module::from_ptx/Graph::instantiate capture `Arc<Context>` (taking `&Arc<Context>` or resolving the current context) so the context strictly outlives every child handle; skip the destroy call in Drop when the owning context is already gone.
- [x] **[correctness] oxicuda-driver** — `crates/oxicuda-driver/src/ffi.rs:729`
  - **Defect:** ~30 CUdevice_attribute discriminants are wrong — public device-property APIs silently return values of unrelated attributes
  - **Impact:** Silently wrong device capability/limit reporting across two crates' public APIs: e.g. supports_tensor_map_access() actually reports dma_buf support (returns true on Linux boxes without Hopper TMA — downstream code may then emit TMA PTX that fails), numa_id() returns a 0/1 fabric-support flag, texture/surface size limits are the limits of different dimensions (potential out-of-range array creation passing validation), and mem_sync_domain_count() returns a boolean of an unrelated MemOp feature.
  - **Fix:** Renumber every listed variant to the cuda.h values, remove the bogus duplicates (AccessPolicyMaxWindowSize=111, MaxTexture1DMipmappedWidth2=52, MaxTimelineSemaphoreInteropSupported=129, MemSyncDomainSupported=130), relabel Reserved92-94 (deprecated STREAM_MEM_OPS_V1 attrs), and regenerate the assertion tests from cuda.h rather than from the enum itself.
- [x] **[correctness] oxicuda-driver** — `crates/oxicuda-driver/src/ffi_launch.rs:22`
  - **Defect:** Every CuLaunchAttributeId discriminant is wrong vs CUlaunchAttributeID — ClusterDimension(2) is actually CU_LAUNCH_ATTRIBUTE_COOPERATIVE, silently mis-configuring cuLaunchKernelEx
  - **Impact:** Any cuLaunchKernelEx launch using these attributes does something different from what the caller requested: cluster-dimension requests become cooperative-launch flags (launch fails with INVALID_VALUE or runs as a grid-sync cooperative kernel), priority requests become programmatic stream serialization (silent scheduling/synchronization changes), mem-sync-domain settings become priorities — silent wrong execution semantics on valid input.
  - **Fix:** Renumber the enum to match CUlaunchAttributeID exactly (IGNORE=0, ACCESS_POLICY_WINDOW=1, COOPERATIVE=2, SYNCHRONIZATION_POLICY=3, CLUSTER_DIMENSION=4, CLUSTER_SCHEDULING_POLICY_PREFERENCE=5, PROGRAMMATIC_STREAM_SERIALIZATION=6, PROGRAMMATIC_EVENT=7, PRIORITY=8, MEM_SYNC_DOMAIN_MAP=9, MEM_SYNC_DOMAIN=10, LAUNCH_COMPLETION_EVENT=12, DEVICE_UPDATABLE_KERNEL_NODE=13) and add a test that asserts the numeric values against cuda.h.
- [x] **[memory-safety] oxicuda-driver** — `crates/oxicuda-driver/src/loader.rs:813`
  - **Defect:** cuMemcpyBatchAsync loaded with a fabricated 6-parameter signature — real CUDA 12.8 export takes 9 parameters; any call on a 12.8+ driver is UB
  - **Impact:** On machines with an r570+/CUDA-12.8 driver, any caller that follows the field's own documentation ('issues count asynchronous memory copies in a single driver call') invokes the driver with mismatched arguments: garbage attrs/attrsIdxs pointers are dereferenced inside libcuda — process crash or silent memory corruption in a production service after a routine driver upgrade.
  - **Fix:** Correct the extern signature to the 9-parameter CUDA 12.8 prototype (dsts/srcs as *mut CUdeviceptr, count as usize, attrs *const CUmemcpyAttributes, attrs_idxs *const usize, num_attrs usize, fail_idx *mut usize, stream) and add the CUmemcpyAttributes descriptor struct, or remove the field until the API is actually wrapped.
- [ ] **[memory-safety] oxicuda-driver** — `crates/oxicuda-driver/src/module.rs:566` ⏸️DEFERRED (soundness batch)
  - **Defect:** Function is Copy with no lifetime tie to Module — safe occupancy/attribute methods pass a dangling CUfunction to the driver after cuModuleUnload
  - **Impact:** Use-after-free of a driver-internal structure reachable from safe Rust: process crash or garbage attribute/occupancy values that silently mis-tune kernel launch configurations in production.
  - **Fix:** Give Function a lifetime (`Function<'m>` with `PhantomData<&'m Module>`) or have it hold `Arc<ModuleInner>` so the module cannot be unloaded while any Function exists; remove `Copy` if a lifetime is added.
- [ ] ⏸️DEFERRED (cascading) **[correctness] oxicuda-driver** — `crates/oxicuda-driver/src/stream.rs:59` _(×2 auditors)_ — _fix was implemented then reverted: making `Stream::new` honour `ctx` exposed a latent bug in the umbrella `oxicuda` backend, which passes a throwaway "token" context to `BlasHandle` while running kernels in the primary context (previously masked because `Stream::new` ignored `ctx`). Must be fixed together with the umbrella backend's context wiring — see the deferred soundness batch._
  - **Defect:** Stream::new never makes the passed Context current — the stream is created in whatever context is current, so the Arc<Context> tether can guard the WRONG context
  - **Impact:** In any multi-context/multi-GPU program the stream can be silently bound to the wrong device (wrong-GPU execution, cross-context errors) and the lifetime guarantee the type advertises does not hold, re-opening the stale-handle destroy hazard the Arc was added to prevent.
  - **Fix:** In Stream::new/with_priority, save the current context, call `ctx.set_current()?` before cuStreamCreate, and restore afterward (or verify via cuStreamGetCtx that the created stream belongs to `ctx` and error otherwise).
- [x] **[resource-leak] oxicuda-driver** — `crates/oxicuda-driver/src/stream_ordered_alloc.rs:417`
  - **Defect:** StreamMemoryPool has no Drop and never calls cuMemPoolDestroy — every StreamMemoryPool::new() permanently leaks a driver-side memory pool
  - **Impact:** Each pool created in a create/drop cycle (e.g. per-request or per-batch pool churn in a long-running service) leaks a CUmemoryPool driver object and any memory the driver keeps reserved under it, accumulating GPU resource exhaustion until context teardown or process exit.
  - **Fix:** Add an `owned: bool` field (true for `new`, false for `default_pool`); implement `Drop` that, when owned and handle != 0, calls `api.cu_mem_pool_destroy(CUmemoryPool(handle as *mut c_void))` and warn-logs failures.
- [x] **[correctness] oxicuda-infer** — `crates/oxicuda-infer/src/cache/kv_cache.rs:231`
  - **Defect:** PagedKvCache::dec_ref on an already-free block pushes a duplicate free-list entry - double-allocation of the same KV block and silent cross-sequence corruption
  - **Impact:** In a serving deployment using prefix sharing, one refcounting off-by-one converts to silent KV-cache aliasing between requests: wrong attention outputs and potential cross-request/cross-user data leakage through shared K/V state - the classic double-free-to-corruption escalation, undetected.
  - **Fix:** Only push to free_list on the 1->0 transition (`if *rc > 0 { *rc -= 1; if *rc == 0 { ...push... } }`), return Err/debug-assert on dec_ref of an rc==0 block and on inc_ref of a free block, and bounds-check id against n_blocks returning Err instead of indexing.
- [x] **[resource-leak] oxicuda-infer** — `crates/oxicuda-infer/src/cache/prefix_cache.rs:129`
  - **Defect:** PrefixCache::insert discards evict_lru()'s returned block IDs - evicted prefixes' KV blocks leak permanently
  - **Impact:** Monotonic KV-cache block exhaustion in long-running inference servers with prefix caching enabled: allocatable blocks shrink with every eviction until alloc_blocks returns BlockAllocFailed for all new sequences - a slow-burn production outage.
  - **Fix:** Change insert() to return the evicted IDs (e.g. `-> Option<Vec<BlockId>>` or accept `&mut PagedKvCache` and dec_ref internally) so the caller can release the blocks; add a #[must_use] on evict_lru().
- [x] **[error-handling] oxicuda-infer** — `crates/oxicuda-infer/src/sampling/top_p.rs:63`
  - **Defect:** top_p_filter panics via .expect("no NaN") on fully-masked (all -inf) or +inf-containing logits despite its documented Err contract
  - **Impact:** A single fully-masked logits row (empty grammar mask, aggressive logit-bias, or an inf produced by upstream scaling) crashes the whole inference server process instead of returning the documented SamplingError.
  - **Fix:** After computing probs, check `probs.iter().any(f32::is_nan)` (or check for `!max.is_finite()`) and return Err(InferError::SamplingError/NanLogits); use total_cmp for the sort so it can never panic.
- [x] **[concurrency] oxicuda-launch** — `crates/oxicuda-launch/src/async_launch.rs:311` _(×2 auditors)_
  - **Defect:** LaunchCompletion/TimedLaunchCompletion futures hang forever under the default Yield (and BackoffMicros) poll strategy: the waker is only scheduled once
  - **Impact:** Production async pipelines using `AsyncKernel::launch_async(...).await` with default config deadlock permanently on any kernel longer than a few microseconds; timeouts configured via AsyncLaunchConfig::with_timeout never trigger because poll is never re-entered. Spin mode instead degrades the host with one OS thread per poll iteration.
  - **Fix:** Make the poller thread loop (query-able via a shared flag/Event clone) and re-wake until completion, or respawn the poller on every Pending poll for all strategies (accepting the thread-per-wake cost), and add a wall-clock re-poll for timeout enforcement.
- [x] **[correctness] oxicuda-launch** — `crates/oxicuda-launch/src/cluster.rs:213`
  - **Defect:** cluster_launch silently discards the cluster dimensions — never passes them to the driver and never checks sm_90 support
  - **Impact:** On Hopper+ hardware, kernels written to rely on cluster grouping (distributed shared memory, cluster.sync(), %cluster_ctarank-based partitioning) run with an implicit 1x1x1 cluster and silently compute wrong results; on sm_86 the API reports success while providing none of the advertised semantics, masking a hard hardware prerequisite.
  - **Fix:** Use `cu_launch_kernel_ex` with a CUlaunchConfig carrying CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION when available; return CudaError::NotSupported when the symbol is absent or the device compute capability is < 9.0.
- [x] **[correctness] oxicuda-launch** — `crates/oxicuda-launch/src/cooperative.rs:122` _(×3 auditors)_
  - **Defect:** CooperativeLaunch::launch never calls cuLaunchCooperativeKernel — silently launches non-cooperatively, so grid-wide sync kernels deadlock or corrupt results
  - **Impact:** Any kernel using grid-wide synchronization (iterative solvers, multi-pass reductions) launched through this advertised cooperative API can deadlock the GPU (blocks waiting at a grid barrier for blocks that will never be scheduled) or, when the barrier degrades to a spin on a global counter, silently produce partially-synchronized, corrupted numeric results in production.
  - **Fix:** Call `(driver.cu_launch_cooperative_kernel)(...)` (already in the loaded function table) or delegate to `oxicuda_driver::cooperative_launch`; delete the false comment.
- [x] **[numeric] oxicuda-launch** — `crates/oxicuda-launch/src/grid.rs:174` _(×3 auditors)_
  - **Defect:** auto_grid_for/auto_grid_2d truncate usize element counts to u32, silently launching too few blocks for n >= 2^32
  - **Impact:** Silent data corruption: large-buffer workloads get partially processed with a success return; downstream consumers (reductions, transforms) read stale/uninitialized regions with no diagnostic.
  - **Fix:** Compute the grid in u64 (`(n as u64).div_ceil(block_size as u64)`), return an error if the result exceeds the device max grid dim (or split across grid.y), and only then narrow to u32.
- [ ] **[memory-safety] oxicuda-launch** — `crates/oxicuda-launch/src/named_args.rs:120` _(×2 auditors)_ ⏸️DEFERRED (soundness batch)
  - **Defect:** Kernel-argument pointer APIs are safe but lifetime-unchecked: ArgBuilder::add, pub-field NamedArgEntry (Send+Sync), and DeviceLaunchConfig store raw arg pointers that cuLaunchKernel later dereferences
  - **Impact:** Freed-memory reads at launch time turn into garbage kernel parameters (including pointer args) → GPU memory corruption or crashes triggered by safe Rust, and the corruption is silent when the garbage happens to be a mapped address.
  - **Fix:** Give ArgBuilder a lifetime parameter tied to the added references (ArgBuilder<'a> with PhantomData<&'a ()>), make NamedArgEntry construction unsafe (private fields + unsafe constructor), and drop the unsafe Send/Sync impls or justify them with owned storage.
- [x] **[correctness] oxicuda-memory** — `crates/oxicuda-memory/src/copy.rs:206` _(×4 auditors)_
  - **Defect:** copy_dtod_async silently ignores the stream and issues a legacy-stream synchronous copy — unordered vs non-blocking streams and serializes the device
  - **Impact:** Wrong results on valid input in any pipeline that enqueues produce-kernel → copy_dtod_async → consume-kernel on one non-blocking stream (the standard oxicuda Stream); plus loss of async overlap for all callers.
  - **Fix:** Call api.cu_memcpy_dtod_async_v2 (already loaded, Option-gated) with stream.raw(); return NotSupported if the symbol is absent.
- [ ] **[memory-safety] oxicuda-memory** — `crates/oxicuda-memory/src/unified.rs:142` ⏸️DEFERRED (soundness batch)
  - **Defect:** UnifiedBuffer::as_slice/as_mut_slice and MappedBuffer host slices are safe fns racing with concurrent GPU access (data race UB from safe code)
  - **Impact:** Torn reads / silently corrupted host-side values whenever unified or zero-copy memory is read while a kernel is in flight; formal UB the compiler may miscompile around.
  - **Fix:** Either make the slice accessors `unsafe fn` with the documented contract, or gate them behind a synchronization token (e.g. a method that takes &Stream and synchronizes before returning the slice).
- [ ] **[memory-safety] oxicuda-memory** — `crates/oxicuda-memory/src/zero_copy.rs:153` ⏸️DEFERRED (soundness batch)
  - **Defect:** Host/GPU data race constructible entirely from safe code: MappedBuffer::as_host_slice and UnifiedBuffer::as_slice are safe fns whose validity depends on 'no concurrent GPU access', while Kernel::launch is also safe
  - **Impact:** Undefined behavior (host/device data race on the same physical memory) reachable from safe Rust; manifests as torn/corrupt values read by the CPU in zero-copy and unified-memory workflows.
  - **Fix:** Make the slice accessors `unsafe fn`, or gate them behind a synchronization token (e.g. require &Stream and internally cuStreamSynchronize before returning the slice).
- [x] **[concurrency] oxicuda-ptx** — `crates/oxicuda-ptx/src/cache.rs:142`
  - **Defect:** PTX disk cache uses non-atomic direct writes, producing torn/interleaved files under concurrency
  - **Impact:** Concurrent oxicuda processes corrupt each other's PTX cache entries; a torn file is later read back as a cache hit and fails JIT compilation, turning a warm-cache launch into a hard error on valid input.
  - **Fix:** Write to a unique temp file in the same directory (e.g. `{final}.{pid}.{tid}.tmp`), fsync, then `std::fs::rename` to the final path (atomic on the same filesystem). Apply the same tmp+rename pattern to ResultDb::flush and PersistentTuneCache::save_at.
- [x] **[correctness] oxicuda-ptx** — `crates/oxicuda-ptx/src/ir/instruction_emit.rs:719`
  - **Defect:** wmma.mma is emitted with ONE brace group and ONE layout/type modifier — PTX ISA requires four operand groups (d,a,b,c), two layouts (alayout.blayout) and two types (dtype.ctype); the whole WMMA path cannot assemble
  - **Impact:** Every kernel using the WMMA builder family (wmma_load_a/b, wmma_mma_sync_*, wmma_store_d) produces PTX that ptxas rejects at JIT — the advertised Volta/Turing tensor-core path is entirely non-functional despite passing all unit tests.
  - **Fix:** Change Instruction::Wmma to carry separate d/a/b/c register vectors plus (alayout, blayout, dtype, ctype); emit `wmma.mma.sync.aligned.{alayout}.{blayout}{shape}{dtype}{ctype} {d}, {a}, {b}, {c};` and reorder load/store emission to layout-then-shape.
- [x] **[correctness] oxicuda-ptx** — `crates/oxicuda-ptx/src/tensor_core/mma.rs:256`
  - **Defect:** MMA fragment register counts ignore element type — TF32/INT8/FP8 counts are wrong per PTX ISA, and validate() accepts the nonexistent m16n8k32.f16/bf16 instruction
  - **Impact:** Every TF32, INT8 (m16n8k16/k32) and FP8 tensor-core kernel built via these documented APIs produces PTX with wrong operand-vector arity (ptxas rejection at JIT), or garbage math if a caller independently packs data to the wrong fragment layout; the TF32 path is reachable on the sm_86 hardware this stack ships on.
  - **Fix:** Make regs_per_thread_a/b depend on (shape, element type): count = M*K*elem_bits/(32*32) for A and K*N*elem_bits/(32*32) for B; fix the three builder wrappers and their doc comments; remove F16/BF16 from the M16N8K32 arm of validate_shape_types and delete/redirect mma_m16n8k32_f16_f32.
- [x] **[correctness] oxicuda-ptx** — `crates/oxicuda-ptx/src/builder/body_builder/mod.rs:1440`
  - **Defect:** wgmma_mma_async_m64n128k16_f16 emits the literal placeholder text "{...}" as the destination register list — guaranteed ptxas syntax error
  - **Impact:** Any caller of this documented Hopper WGMMA helper produces PTX that can never assemble; the failure surfaces only at JIT on sm_90 deployment targets.
  - **Fix:** Delete this method or reimplement it as a thin wrapper over tensor_core_ops::wgmma_mma_async (which allocates the 64 accumulator registers and emits the structured Instruction::Wgmma).
- [x] **[correctness] oxicuda-ptx** — `crates/oxicuda-ptx/src/ir/operand.rs:63`
  - **Defect:** Float immediates are printed with Rust Display — NaN prints as "NaN" and infinity as "inf", which PTX cannot parse; hex bit-pattern form (0f/0d) is never used
  - **Impact:** Kernels containing a NaN/Inf float immediate — directly requested or produced by the crate's own constant folder from finite constants — generate PTX that fails cuModuleLoadDataEx at runtime; every downstream consumer of the affected kernel loses the launch.
  - **Fix:** Emit F32 immediates as format!("0f{:08X}", v.to_bits()) and F64 as format!("0d{:016X}", v.to_bits()) unconditionally (matching mov_imm_f64), which is lossless and handles NaN/Inf/-0.0/subnormals.
- [x] **[correctness] oxicuda-ptx** — `crates/oxicuda-ptx/src/emit/printer.rs:105`
  - **Defect:** printer.rs still emits .maxntid INSIDE the kernel body with a trailing semicolon — the exact placement commit 4f058d2 proved ptxas rejects, fixed in templates/KernelBuilder but not here
  - **Impact:** Any downstream user emitting IR through the documented printer API with launch bounds set gets PTX that fails cuModuleLoadDataEx — the same 'every kernel unassemblable' failure mode the 4f058d2 commit fixed elsewhere.
  - **Fix:** Move the .maxntid writeln above the `writeln!(out, "{{")` in all three printer functions and drop the trailing semicolon, mirroring kernel_builder.rs:188-190; add the same placement regression test.
- [x] **[memory-safety] oxicuda-sparse** — `crates/oxicuda-sparse/src/ops/spmv.rs:261`
  - **Defect:** CSR column indices are never range-checked (host or device): a col_idx >= cols or negative i32 makes SpMV/SpMM read device memory out of bounds
  - **Impact:** A single corrupted or miscomputed column index (e.g. from an external matrix file) accepted by the validating constructor causes GPU out-of-bounds reads — silently wrong results at best, a context-killing fault (taking down all in-flight GPU work) at worst.
  - **Fix:** Add an O(nnz) host-side check in CsrMatrix::from_host / CooMatrix::from_host that every index is in [0, cols) (resp. rows), and mark from_device as the explicit unchecked escape hatch. Optionally emit a device-side bounds guard in debug builds.
- [x] **[error-handling] oxicuda-train** — `crates/oxicuda-train/src/zero.rs:168`
  - **Defect:** ZeroOptimizer::new documents 'Returns InvalidRank if the config is invalid' but returns Self and panics via .expect() instead
  - **Impact:** A wrong RANK/WORLD_SIZE environment value (the most common launcher misconfiguration in distributed training) crashes the process with a panic instead of the documented, handleable TrainError::InvalidRank - and the doc/behavior contradiction misleads downstream error handling.
  - **Fix:** Change the signature to `pub fn new(base: O, config: ZeroConfig) -> TrainResult<Self>` and propagate config.validate()? (mechanical change; call sites already live in Result-returning contexts).

### 🟡 MEDIUM (47)

- [x] **[performance] oxicuda** `backend.rs:481` — Backend gemm/conv2d/attention create a throwaway CUDA context and a fresh BLAS/DNN handle (new stream + capability query) on every operation
- [x] **[performance] oxicuda** `ptx_ops.rs:391` — Umbrella backend launches every elementwise/reduce op with a throwaway cuCtxCreate + cuStreamCreate + full stream synchronize + fresh JIT compile
- [x] **[correctness] oxicuda** `ptx_ops.rs:195` — PTX disk-cache key omits the generator/crate version — stale kernels persist across library upgrades
- [x] **[api-robustness] oxicuda-ann** `serializer.rs:417` — Deserializers pass unvalidated length fields straight to Vec::with_capacity - capacity-overflow panic / OOM abort on corrupt or malicious files
- [x] **[correctness] oxicuda-autotune** `benchmark.rs:366` — benchmark_wallclock provides no GPU synchronization but is used by ConstrainedTuner to time kernels → wrong winners
- [x] **[error-handling] oxicuda-autotune** `config.rs:196` — Config::estimated_registers_per_thread panics (divide-by-zero) when block_size == 0, aborting prune()
- [x] **[concurrency] oxicuda-autotune** `result_db.rs:273` — ResultDb rewrites the whole JSON file on every save with no locking → last-writer-wins / lost updates
- [x] **[error-handling] oxicuda-autotune** `result_db.rs:115` _(×2 auditors)_ — ResultDb::open_at deserializes benchmark results with no value validation; Dispatcher then serves them verbatim
- [x] **[api-robustness] oxicuda-autotune** `search_space.rs:159` — SearchSpace::prune never validates block_size against device limits (max 1024 threads, multiple of 32)
- [x] **[performance] oxicuda-blas** `axpy.rs:77` — Systemic JIT-per-call: every non-GEMM BLAS op regenerates PTX and cuModuleLoadData-compiles a fresh module on every invocation — no module cache exists anywhere in the crate
- [x] **[api-robustness] oxicuda-blas** `complex_gemm.rs:145` — complex_gemm maps rows directly to grid.y — any m > 1,048,560 exceeds the 65,535 grid.y hardware limit and the launch fails on valid input
- [x] **[performance] oxicuda-blas** `gemm_api.rs:123` — gemm() constructs a fresh GemmDispatcher per call, so the crate's only compiled-kernel cache is always empty — every GEMM re-JITs its full tiled kernel
- [ ] **[performance] oxicuda-blas** `trmm.rs:153` — Per-call raw cuMemAlloc workspaces plus forced full-stream synchronize in library ops; the existing memory pool is never used by any math crate ⏭️SKIPPED (needs different fix)
- [x] **[error-handling] oxicuda-driver** `context.rs:233` — Context::scoped has no RAII restore guard — a panicking closure leaves this context current (previous context never restored)
- [x] **[api-robustness] oxicuda-driver** `debug.rs:693` — PrintfBuffer::parse_entries pre-allocates Vec::with_capacity(arg_count) from an untrusted u32 — a 13-byte malformed buffer can abort the process
- [x] **[concurrency] oxicuda-driver** `ffi.rs:46` — define_handle! blanket-implements Send+Sync for every opaque CUDA handle, silently making Context/Stream/Event/Module auto-Sync and voiding the crate's documented thread-safety model; DevicePool's SAFETY comment is also factually wrong
- [x] **[correctness] oxicuda-driver** `graph.rs:578` — GraphExec::launch reports success while executing nothing: every kernel/memcpy/memset node is lowered to cuGraphAddEmptyNode
- [x] **[error-handling] oxicuda-driver** `loader.rs:1686` — try_driver() discards all driver-load diagnostics (which symbol failed, cuInit error code) and permanently caches transient cuInit failures
- [x] **[concurrency] oxicuda-driver** `multi_gpu.rs:70` — unsafe impl Send for Stream and unsafe impl Send+Sync for DevicePool smuggle Arc<Context> across threads, violating the crate's own Context threading model; the DevicePool SAFETY comment is factually wrong
- [x] **[correctness] oxicuda-driver** `stream_ordered_alloc.rs:515` — Driver-backed StreamMemoryPool hands out synthetic CPU-model addresses typed as CUdeviceptr — real cuMemAllocAsync binding is dead code
- [x] **[error-handling] oxicuda-infer** `kv_cache.rs:81` — KvBlock::append / PagedKvCache::append_token panic in copy_from_slice on wrong k/v vector length - the Result-returning API validates layer count but never element length
- [x] **[performance] oxicuda-infer** `top_p.rs:97` — top_p_filter builds an O(n^2) 'nucleus_set' with nested full-vocab loops and then explicitly discards it - dead quadratic work on the per-token sampling hot path
- [x] **[numeric] oxicuda-launch** `grid.rs:99` — Dim3::total() multiplies three u32s without widening — panics in debug / wraps in release for valid large grids, feeding the cooperative gate and thread accounting
- [x] **[numeric] oxicuda-launch** `telemetry.rs:64` — estimate_occupancy uses wrong hardware constants (sm_86 max warps 64 vs real 48; sm_80/sm_89 max blocks/SM 16 vs real 32/24; sm_86 smem/SM 163,840 vs real 102,400)
- [x] **[memory-safety] oxicuda-memory** `host_buffer.rs:130` — PinnedBuffer/UnifiedBuffer/MappedBuffer::alloc expose uninitialized memory as initialized &[T] via safe accessors
- [x] **[resource-leak] oxicuda-memory** `host_registered.rs:315` — register() leaks the host-memory registration (pinned pages) if cuMemHostGetDevicePointer fails after a successful cuMemHostRegister
- [x] **[correctness] oxicuda-memory** `managed_hints.rs:382` — MigrationPolicy::PreferHost silently sets preferred location to a GPU device, and PreferDevice ignores its ordinal
- [x] **[correctness] oxicuda-memory** `peer_copy.rs:77` — enable_peer_access/disable_peer_access permanently clobber the calling thread's current CUDA context and may leave a destroyed context current
- [x] **[concurrency] oxicuda-memory** `peer_copy.rs:267` _(×2 auditors)_ — copy_peer_async releases both primary-context retains immediately after enqueuing the async peer copy
- [x] **[concurrency] oxicuda-memory** `pool.rs:87` — MemoryPool is not bound to its device_ordinal or any context: allocations go to the caller's current context, so a pool shared across threads mixes pointers from different devices in one free-bin map
- [x] **[error-handling] oxicuda-peft** `serialize.rs:177` — Unbounded per-tensor length prefix drives Vec::with_capacity, aborting the process on a tiny malformed adapter file
- [x] **[correctness] oxicuda-ptx** `cache.rs:52` — PtxCacheKey omits generator / PTX-ISA / oxicuda version, so stale PTX is served after a codegen upgrade
- [x] **[security] oxicuda-ptx** `cache.rs:227` — PTX cache falls back to a world-writable /tmp dir with predictable filenames (kernel-poisoning vector)
- [x] **[security] oxicuda-ptx** `cache.rs:172` — PTX and tuning caches fall back to a predictable world-writable /tmp path and write via symlink-following fs::write with no permission hardening
- [x] **[correctness] oxicuda-ptx** `instruction_emit.rs:393` — Structured Wgmma emission passes an integer immediate where the PTX ISA requires a predicate register (scale-d), and the FP8 convenience wrappers pair e4m3/e5m2 with K=16 shapes that do not exist
- [x] **[correctness] oxicuda-ptx** `instruction_emit.rs:597` — tex.1d/2d/3d are emitted with the .v4 modifier but a single scalar destination register — PTX requires a 4-register vector destination
- [x] **[correctness] oxicuda-ptx** `instruction_emit.rs:563` — Systemic: Hopper/Blackwell sync & bulk-copy emissions violate PTX ISA grammar (mbarrier.arrive/try_wait missing required destination, elect.sync missing d|p pair, stmatrix.x4 with one source reg, malformed fence.proxy, wrong cp.async.bulk completion, bogus tcgen05.mma)
- [x] **[correctness] oxicuda-ptx** `instruction_emit.rs:470` — MovSpecial always emits mov.u32, but SpecialReg includes the 64-bit %clock64 — reading it generates a width-mismatched instruction
- [x] **[correctness] oxicuda-ptx** `mod.rs:1643` — raw_ptx auto-declares every %f_* named register as 32-bit (.reg .b32), so raw f64/f16 code — including the crate's own quality-gate kernels — gets wrong-width registers
- [x] **[api-robustness] oxicuda-ptx** `validator.rs:488` — validate_ptx hard-errors on >255 distinct VIRTUAL registers (valid PTX — ptxas spills), while missing the checks that matter: duplicate/undefined branch labels, most arch gating, unknown .target silently disables all SM checks
- [x] **[error-handling] oxicuda-quantum** `metrics.rs:21` — Systemic: caller-input assert!/assert_eq! in library functions abort the process on dimension/parameter mismatches instead of returning Err
- [x] **[error-handling] oxicuda-runtime** `device.rs:307` — device_synchronize() discards the cuCtxSynchronize return code and always returns Ok — asynchronous kernel failures are silently swallowed at the canonical error-reporting point
- [x] **[api-robustness] oxicuda-runtime** `memory.rs:373` — memcpy/memcpy_async with MemcpyKind::Default unconditionally issue cuMemcpyHtoD instead of inferring direction, breaking cudaMemcpyDefault semantics for D2H/D2D transfers
- [x] **[resource-leak] oxicuda-runtime** `peer.rs:115` — memcpy_peer/memcpy_peer_async retain both devices' primary contexts on every call, never release them, and ignore the retain return codes (possibly passing NULL contexts to cuMemcpyPeer)
- [x] **[error-handling] oxicuda-signal** `iir.rs:69` — Biquad design constructors (lowpass/highpass/bandpass/peaking_eq) panic via .expect("valid biquad") on invalid Q instead of returning SignalResult
- [x] **[error-handling] oxicuda-solver** `batched.rs:155` — batched_lu / batched_cholesky hardcode BatchedResult { failed_count: 0 } — singular or non-SPD matrices are reported as successfully factorized
- [x] **[api-robustness] oxicuda-sparse** `coo.rs:166` — CooMatrix::to_csr / to_csc index host vectors with unvalidated device-sourced i32 row/col indices — library panic (or wrong CSR) on constructor-accepted input

### ⚪ LOW (17)

- [x] **[concurrency] oxicuda** `distributed.rs:481` — FileStore::add performs a non-atomic read-modify-write on a shared rendezvous counter, losing concurrent updates
- [x] **[correctness] oxicuda** `mixed_precision.rs:317` — AutocastGuard::drop pops the top of the thread-local autocast stack regardless of which guard is being dropped — out-of-order guard drops silently activate the wrong autocast context
- [ ] **[security] oxicuda-autotune** `power_aware.rs:135` — nvidia-smi is executed by relative name, resolved through the ambient PATH ❌DROPPED (false positive)
- [x] **[correctness] oxicuda-blas** `gemm_api.rs:129` — No aliasing detection anywhere in GEMM/level-3: passing C overlapping A or B launches a kernel that concurrently reads and writes the same buffer
- [x] **[performance] oxicuda-blas** `mod.rs:51` — Fixed 256-thread block size hardcoded across all math-crate launches; the driver's occupancy API is never consulted
- [x] **[correctness] oxicuda-driver** `cupti_stubs.rs:60` — CuptiActivityKind discriminants for ConcurrentKernel/Name/Marker/MemoryPool do not match cupti_activity.h despite the doc promising pass-through compatibility
- [x] **[api-robustness] oxicuda-driver** `ffi.rs:117` — CUtexObject/CUsurfObject modeled as pointer-typed handles but cuda.h defines them as unsigned long long values — breaks on any 32-bit target and misrenders in Debug
- [x] **[correctness] oxicuda-driver** `multi_gpu.rs:107` — DevicePool::with_devices stacks N contexts on the creating thread's context stack and never pops them; after pool drop the stack holds destroyed contexts
- [x] **[api-robustness] oxicuda-driver** `stream_ordered_alloc.rs:1139` — stream_alloc() creates a throwaway pool, returns an allocation, then destroys the pool on return — the returned StreamAllocation is orphaned and its bytes unaccounted
- [ ] **[performance] oxicuda-launch** `kernel.rs:115` — Per-launch heap churn: every kernel launch allocates a Vec for parameter pointers and every Kernel construction allocates a fresh name String ⏭️SKIPPED (needs different fix)
- [x] **[error-handling] oxicuda-mamba** `mamba_model.rs:114` — MambaModelWeights::zeros/random panic via .expect on invalid MambaConfig - reachable because MambaConfig fields are all pub, bypassing new()'s validation; vocab_size*d_model multiply is unchecked
- [x] **[api-robustness] oxicuda-memory** `buffer_view.rs:160` — view_as/view_as_mut check size divisibility but never alignment of the base pointer for the target type U
- [ ] **[error-handling] oxicuda-memory** `pool.rs:145` — Mutex-poison and lock-failure swallowing on statistics/trace paths silently desynchronizes pool accounting ❌DROPPED (false positive)
- [x] **[error-handling] oxicuda-ptx** `instruction_emit.rs:658` — Ldmatrix silently maps any unsupported num_fragments to .x1 while still emitting all destination registers — self-inconsistent instruction instead of an error
- [x] **[api-robustness] oxicuda-ptx** `register.rs:116` — declare_named accepts names colliding with allocator-generated registers or PTX special registers, producing duplicate/illegal .reg declarations with no guard
- [x] **[numeric] oxicuda-rand** `alias.rs:47` — AliasTable::new accepts NaN/inf weights (NaN fails the `w < 0.0` check) and silently degrades to a uniform distribution
- [x] **[error-handling] oxicuda-signal** `multilevel.rs:96` — multilevel_forward computes '1 << levels' with unvalidated levels — debug-build panic / release masked-shift for levels >= 64

### ⏸️ Deferred: coordinated `unsafe`-soundness batch (next version bump)

These 8 findings are CONFIRMED-real soundness holes — a safe `fn` exposing an `unsafe` FFI/aliasing contract, so UB is reachable from 100% safe code. The *correct* fix marks the API `unsafe fn` (and/or adds a lifetime). They are grouped and deferred because (a) the blast radius is workspace-wide and (b) marking a *subset* would leave an inconsistent, half-`unsafe` launch/copy API — worse than doing them together. Do them as ONE deliberate breaking-change PR at the next version boundary, gated on the whole-workspace build. Call-site counts measured on branch `0.4.1`.

- [ ] **F006 `oxicuda-launch::Kernel::launch` → `unsafe fn`** — **218 call sites across 63 crates.** The core launch path. `launch!` macro must emit `unsafe {}`. **Non-breaking interim hardening available:** validate arg count + total arg bytes vs `cuFuncGetParamInfo` / PTX `.param` metadata inside `launch()` and return `Err` on mismatch — closes the arity/size-mismatch → OOB case without a signature change (residual same-layout type-confusion still needs the `unsafe` marking).
- [ ] **F009 `oxicuda-memory::register_vec/register_slice/register` → lifetime-bound `RegisteredMemory<'a, T>`** — **~58 sites.** UAF + `&mut` aliasing from safe code (handle outlives/aliases the source `Vec`).
- [ ] **F036 / F037 `UnifiedBuffer::as_slice/as_mut_slice`, `MappedBuffer::as_host_slice` → `unsafe fn`** (or a `&Stream`-synchronised accessor) — host/GPU data race from safe code. Large `as_slice`/`as_host_slice` surface.
- [ ] **F008 `oxicuda-memory::DeviceBuffer::copy_*_async` raw-slice variants → `unsafe fn`** — **0 in-workspace callers** (safe to mark now, but grouped for API coherence). Also record the last stream and `synchronize()` it in `Drop` for owned `DeviceBuffer`/`PinnedBuffer` before free (non-breaking guard).
- [ ] **F034 `oxicuda-launch::ArgBuilder::add` / `NamedArgEntry` → lifetime param + `unsafe` constructor** — **~29 sites.** Raw arg pointers stored past the referent's lifetime; drop the unjustified `Send`/`Sync`.
- [ ] **F004 `oxicuda-driver::cooperative_launch` / `cooperative_launch_multi_device` → `unsafe fn`** — **12 sites.** Driver dereferences caller arg pointers; the `# Safety` contract is already documented, only the marking is missing.
- [ ] **F024 `oxicuda-driver::Function` → `Function<'m>` (or hold `Arc<ModuleInner>`)** — `Function` is `Copy` with no tie to its `Module`; occupancy/attribute methods can pass a dangling `CUfunction` after `cuModuleUnload`. Remove `Copy` if a lifetime is added.
- [x] **F025 `oxicuda-driver::Stream::new`/`with_priority` bind `ctx` current + restore — FIXED (2026-07-07):** implemented, reverted, then re-applied together with the coordinated backend fix — the correct fix (save current ctx → `ctx.set_current()` → create stream → restore) passed the 439-test driver suite, but exposed that the umbrella `oxicuda` backend passed a *throwaway* `Context::new()` "token" to `BlasHandle` while running kernels in the retained **primary** context — previously masked because `Stream::new` ignored `ctx`, so the stream landed in the primary context by accident. With `Stream::new` honouring `ctx`, the handle's stream lived in the token context while kernels compiled/launched in the primary → `CUDA_ERROR_INVALID_HANDLE` at launch. **Both halves were applied together**: `oxicuda/src/backend.rs::handle_context_token` now binds the BLAS/DNN handle to the primary context — via `oxicuda_driver::Context::from_raw_borrowed` on the primary context's raw handle, rather than a throwaway `Context::new()` token — and the `Stream::new`/`with_priority` context-binding fix (`create_stream_in_ctx` in `oxicuda-driver/src/stream.rs`) was re-applied alongside it, resolving the cascading `CUDA_ERROR_INVALID_HANDLE` issue.

**Also still open (not in the soundness batch):**
- [ ] **F058 `oxicuda-blas` `trmm.rs` per-call workspaces** — needs a *real* stream-ordered/pooled workspace path; the originally-recommended `stream_alloc`/`stream_free` route through an unsound CPU-model allocator, so it was not applied.
- [ ] **F102 `oxicuda-launch` `kernel.rs:115` per-launch heap churn** — fixing it changes `KernelArgs::as_param_ptrs`'s public return type (api-breaking); fold into the soundness batch.
- [~] **Alt-backend audit** (oxicuda-vulkan / metal / rocm / levelzero / webgpu) — never ran (session token limit); re-run the alt-backend audit next session.
  - **Status (2026-08-12):** `oxicuda-metal` and `oxicuda-webgpu` audited and fixed on real Apple M3 hardware (Metal 4, macOS, arm64) — see the "Remaining GAP" summary above (~line 29) and `CHANGELOG.md` `[0.5.5]` for the full writeup. `oxicuda-metal`: 401/401 tests passing (`--all-features`), clippy-clean. `oxicuda-webgpu`: 293/293 (`--all-features`), clippy-clean. Both crates also pass 686/686 under `--features gpu-tests` on real hardware, and the whole workspace is clippy-clean at `--workspace --all-targets --all-features` (and at default features) with 38,924/38,924 tests passing at `--all-features` and 37,593/37,593 at default features (2 skipped in each — the pre-existing `#[ignore = "requires GPU"]` CUDA tests in `oxicuda-train`).
  - **Remaining:** `oxicuda-vulkan` / `oxicuda-rocm` / `oxicuda-levelzero` are still entirely un-audited, 0 findings collected — re-run next session.
- [x] **[correctness] oxicuda-dnn — FIXED (2026-07-07):** `mha::multi_head_attention`'s three stub kernels (`generate_qk_gemm_ptx`, `generate_row_softmax_ptx`, `generate_pv_gemm_ptx`) now compute real values — a runtime dot-product loop for QK^T/PV (flat-index `div.u32`/`rem.u32` decomposition into `(row, col, batch_head)`, modeled on `rnn::lstm`'s loop-carried-accumulator idiom) and a serial 3-pass stable softmax (max, `ex2.approx` exp-sum, `rcp.approx` normalize) per row. Also fixed a buffer-overflow latent in the same function: it previously reused `output.ptr` (sized `[total_heads, seq_len, head_dim]`) to hold the `[total_heads, seq_len, seq_len]` score matrix, which overflows the output allocation whenever `seq_len != head_dim`; a dedicated scratch `DeviceBuffer` now backs the score matrix, and the QK^T/PV grid sizes were corrected to the actual output-element counts (they were previously sized off the wrong dimension products). `validate_mha_shapes` now also rejects `Q`/`K` sequence-length mismatches, since this naive (non-flash) path doesn't support cross-attention shapes. Two on-device numeric regression tests added in `gpu_tests/attn.rs` (`mha_pipeline_numeric_no_mask`/`_with_mask`, both with `seq_len != head_dim` and batch/heads > 1) confirm the fix against an f64 CPU oracle. **Caveat: `generate_scale_mask_ptx` (the scale+mask step between QK^T and softmax) is hardcoded f32 regardless of its `T` type parameter — untouched here since it predates and is outside this bug's scope, but it means the f64 path through `multi_head_attention` is still broken; only f32 was fixed/tested.**
- [x] **[correctness] oxicuda-blas / oxicuda-ptx — FIXED (2026-07-07):** `CausalSoftmaxTemplate::generate` now takes an explicit `seq_len` kernel parameter and computes `row_in_seq = row % seq_len` (via `rem.u32`) before deriving the live-column count, so the causal boundary resets at every `seq_len`-row matrix instead of saturating to "fully unmasked" past the first one. `oxicuda_blas::reduction::causal_softmax` threads `seq_len` through its public signature and CPU oracle; a new on-device regression test (`device_matches_reference_for_batched_matrices`, 3 stacked `seq_len=5` matrices) confirms the last block's row 0 is masked down to one-hot rather than fully unmasked. Note: this GPU kernel had no callers anywhere else in the workspace at the time of the fix (confirmed by search), so no other call site needed updating.
- [x] **Trustformers-reported upstream GEMM bugs — triaged and closed (2026-07-07):** trustformers (downstream consumer of oxicuda 0.4.0) reported 3 bugs it had worked around locally: (1) batched-GEMM launch-tuple corruption, (2) GEMM ignoring transpose flags / `lda`, (3) the MHA bug recorded as still-open above. Verified against current HEAD this session: **(1) and (2) are already fixed by commit `228b5ad` (2026-07-06)** — `crates/oxicuda-blas/src/batched/strided_gemm.rs::gemm_strided_batched` now validates `trans_a`/`trans_b` and `lda/ldb/ldc/ldd` and honestly rejects (`Err(BlasError::UnsupportedOperation)`) transposed or non-tightly-packed input instead of launching the historical corrupted 17-field-tuple-to-8-parameter-kernel launch; and `crates/oxicuda-blas/src/level3/gemm_api.rs::gemm_impl` correctly threads `trans_a`/`trans_b` into the `GemmProblem` handed to the dispatcher and rejects non-row-major / non-tightly-packed (`ld != cols`) operands instead of silently mis-addressing them. These correspond to the `[x]` CRITICAL findings above (`dispatch.rs` transpose, `strided_gemm.rs` arg-tuple); no further action needed.

---

## Design Principles

- **Pure Rust**: Zero C/Fortran dependencies. No CUDA SDK, no nvcc, no C toolchain.
- **Minimal Dependencies**: Only `libloading`, `thiserror`, `num-complex`, `half` (optional), `serde` (optional).
- **Runtime-only GPU**: `libcuda.so` / `nvcuda.dll` loaded dynamically at runtime via `libloading`.
- **No Unwrap**: All fallible operations return `Result<T, E>`. No `unwrap()` or `expect()` in library code.
- **No Warnings**: Zero clippy warnings, zero compiler warnings across the entire workspace.
- **Workspace Inheritance**: Single version source of truth via `Cargo.toml` workspace.

See [oxicuda-estimation.md](oxicuda-estimation.md) for detailed project estimation
(original estimate: 1.6M-3.1M SLoC; actual: 42K SLoC core implementation).

---

## Vol.1: Foundation Architecture & Driver Layer [COMPLETE]

### oxicuda-driver (24 files, 8,970 SLoC)
- [x] FFI type definitions (ffi.rs) -- CUdevice, CUcontext, CUstream, CUmodule, CUfunction
- [x] Error handling (error.rs) -- CudaError with ~100 CUDA result code variants
- [x] Dynamic loader (loader.rs) -- libcuda.so / nvcuda.dll via libloading, no build-time SDK
- [x] Device management (device.rs) -- enumeration, attributes, compute capability queries
- [x] Context management (context.rs) -- RAII CudaContext with push/pop stack semantics
- [x] Stream management (stream.rs) -- async streams, synchronization, callback support
- [x] Event management (event.rs) -- timing, inter-stream synchronization, elapsed time
- [x] Module/PTX loading (module.rs) -- load PTX/cubin, function handle extraction
- [x] Occupancy API (occupancy.rs) -- max active blocks, suggested block size
- [x] Multi-GPU context management (multi_gpu.rs) -- DevicePool with per-device context pool, round-robin scheduling, best_available_device selection
- [x] Graph API (graph.rs) -- Graph, GraphNode, GraphExec, StreamCapture (cudaGraph equivalent)
- [x] Driver version queries (device.rs) -- cuDriverGetVersion
- [x] Peer-to-peer access (device.rs) -- cuDeviceCanAccessPeer
- [x] Primary context management (primary_context.rs) -- cuDevicePrimaryCtxRetain/Release

### oxicuda-memory (17 files, 4,081 SLoC)
- [x] DeviceBuffer<T> (device_buffer.rs) -- typed GPU allocation, Send + Sync, RAII free
- [x] PinnedBuffer<T> (host_buffer.rs) -- page-locked host memory for async transfers
- [x] Copy operations (copy.rs) -- H2D, D2H, D2D, async variants with stream ordering
- [x] Unified memory (unified.rs) -- managed memory with automatic page migration
- [x] Memory pool (pool.rs) -- feature-gated async memory pool (CUDA 11.2+)
- [x] Zero-copy memory (zero_copy.rs) -- host-mapped device-accessible memory
- [x] Async copy operations (copy.rs) -- copy_htod_async_raw, copy_dtoh_async_raw, copy_dtod_async with stream ordering
- [x] Multi-GPU peer copy (peer_copy.rs) -- can_access_peer, enable/disable_peer_access, copy_peer, copy_peer_async
- [x] Async memory pool enhancements (pool.rs) -- PoolStats (allocated/peak/count), trim(), set_threshold()
- [x] Virtual memory management (virtual_memory.rs) -- VirtualAddressRange, PhysicalAllocation, VirtualMemoryManager
- [x] 2D/3D memory copy (copy_2d3d.rs) -- Memcpy2DParams, Memcpy3DParams, copy_2d/3d functions
- [x] Memory advice / prefetch hints (memory_info.rs) -- cuMemAdvise, cuMemPrefetchAsync
- [x] Buffer views and reinterpret cast (buffer_view.rs) -- type-safe buffer reinterpretation
- [x] Memory usage query (memory_info.rs) -- cuMemGetInfo (free/total VRAM)
  - [x] Memory pool statistics (pool_stats.rs) -- AllocationHistogram, FragmentationMetrics, PoolStatsTracker
  - [x] Managed memory hints (managed_hints.rs) -- ManagedMemoryHints, MigrationPolicy, PrefetchPlan

### oxicuda-launch (15 files, 4,161 SLoC)
- [x] Dim3 + grid helpers (grid.rs) -- 1D/2D/3D dimension types, grid size calculators
- [x] LaunchParams builder (params.rs) -- type-safe kernel launch configuration
- [x] Kernel + KernelArgs trait (kernel.rs) -- compile-time argument type validation
- [x] launch! macro (macros.rs) -- CUDA-style <<<grid, block, smem, stream>>> syntax
- [x] Launch bounds validation (error.rs, params.rs) -- LaunchParams::validate() checks block/grid/shared memory against device limits
- [x] Occupancy-based auto grid sizing (grid.rs) -- auto_grid_for(), auto_grid_2d() using occupancy API
- [x] Cooperative launch (cooperative.rs) -- CooperativeLaunch with max_active_blocks, optimal_block_size
- [x] Graph-based launch (graph_launch.rs) -- GraphLaunchCapture for launch recording
- [x] Cluster launch (cluster.rs) -- ClusterDim, ClusterLaunchParams (Hopper+)
- [x] Multi-stream launch (multi_stream.rs) -- multi_stream_launch across streams
- [x] Named kernel arguments (named_args.rs) -- NamedKernelArgs trait, ArgBuilder

### oxicuda-nvrtc (5 files, 642 SLoC) [NEW in 0.5.1]
- [x] Dynamic loader (loader.rs) -- libnvrtc.so / nvrtc64_*.dll via libloading, no build-time SDK; process-wide OnceLock function table
- [x] Error handling (error.rs) -- NvrtcError: Unavailable (names every candidate tried), MissingSymbol, NotSupported, Call, Compilation (carries the compiler log), InteriorNul, InvalidUtf8
- [x] Program lifecycle (program.rs) -- Program RAII compilation unit (Send), new/with_headers, compile, log
- [x] PTX + CUBIN output (program.rs, ptx.rs) -- Ptx owned NUL-terminated buffer; as_str() feeds oxicuda_driver::Module::from_ptx directly; as_bytes_with_nul, into_bytes_with_nul
- [x] In-memory headers (program.rs) -- Header<'a>, with_headers for #include resolution without touching the filesystem
- [x] C++ name expressions (program.rs) -- add_name_expression / lowered_name for mangled template kernel lookup
- [x] Top-level queries (loader.rs) -- is_available(), version() -> NvrtcVersion, supported_archs()
- [x] Graceful degradation -- no NVRTC yields typed Unavailable rather than a panic; optional symbols (cubin, name expressions, supported-arch) yield NotSupported instead of failing the whole library load
- [x] Umbrella integration -- re-exported as `oxicuda::nvrtc` behind the off-by-default `nvrtc` feature; `HAS_NVRTC` capability constant
- [ ] Add `nvrtc` to the umbrella's `full` feature (currently omitted, so `--features full` does not build it)

---

## Vol.2: PTX Code Generator & Autotuner Engine [COMPLETE]

### oxicuda-ptx (48 files, 24,828 SLoC)
- [x] PTX IR type system (ir/types.rs) -- .f16, .bf16, .f32, .f64, .b8/.b16/.b32/.b64, .pred
- [x] Register allocator (ir/register.rs) -- virtual register allocation, spill tracking
- [x] Instruction representation (ir/instruction.rs) -- typed PTX instruction encoding
- [x] Operand types (ir/operand.rs) -- register, immediate, address, predicate operands
- [x] Basic block (ir/block.rs) -- labeled blocks with instruction sequences
- [x] Function definition (ir/function.rs) -- kernel/device function with parameters
- [x] Module definition (ir/module.rs) -- .version, .target, global declarations
- [x] KernelBuilder DSL (builder/kernel_builder.rs) -- fluent API for kernel construction
- [x] BodyBuilder (builder/body_builder.rs) -- instruction-level builder within kernel body
- [x] Architecture rules (arch.rs) -- SM 7.5 through SM 10.0 capability tables
- [x] Tensor Core generation -- WMMA (tensor_core/wmma.rs), MMA (tensor_core/mma.rs), WGMMA (tensor_core/wgmma.rs)
- [x] PTX text emitter (emit/printer.rs) -- IR to PTX text serialization
- [x] PTX validator (emit/validator.rs) -- structural and semantic validation
- [x] Disk-based PTX cache (cache.rs) -- hash-keyed file cache for compiled PTX
- [x] Atomic operations (ir/instruction.rs) -- Atom, AtomCas, Red instructions; AtomOp enum (Add, Min, Max, Inc, Dec, And, Or, Xor, Exch); 20 BodyBuilder methods
- [x] Bit manipulation (ir/instruction.rs) -- Brev, Clz, Popc, Bfind, Bfe, Bfi instructions; 21+ BodyBuilder methods
- [x] Special math (ir/instruction.rs) -- Rcp, Rsqrt, Sqrt, Ex2, Lg2, Sin, Cos instructions with rounding modes
- [x] Register pressure analysis (analysis/register_pressure.rs) -- peak tracking, spill risk, occupancy estimation
- [x] Dead code elimination (analysis/dead_code.rs) -- fixed-point DCE with liveness analysis
- [x] GEMM template (templates/gemm.rs) -- parameterized GEMM kernel generation
- [x] Elementwise template (templates/elementwise.rs) -- unary/binary elementwise ops
- [x] Reduction template (templates/reduction.rs) -- parallel reduction kernels
- [x] Softmax template (templates/softmax.rs) -- numerically stable softmax kernel
- [x] Scan/prefix-sum template (templates/scan.rs) -- Blelloch work-efficient scan with inclusive/exclusive, sum/product/min/max ops
- [x] Transpose template (templates/transpose.rs) -- coalesced shared-memory transpose with bank-conflict-free padding
- [x] Attention template (templates/attention.rs) -- FlashAttention-style fused attention kernel
- [x] Batch normalization template (templates/batch_norm.rs) -- training + inference BN kernels
- [x] MoE template (templates/moe.rs) -- top-k gating, permute, expert GEMM, unpermute
- [x] Convolution template (templates/convolution.rs) -- im2col, direct conv, 1x1 optimized, backward data/filter
- [x] Video instructions: dp4a, dp2a (ir/instruction.rs, builder/body_builder.rs)
- [x] PTX-level loop unrolling -- pragma unroll, manual unroll in builder
- [x] Integer multiply-add (ir/instruction.rs, builder/body_builder.rs) -- mad.lo, mad.hi, mad.wide instructions
- [x] Texture/surface instructions (ir/instruction.rs, ir/texture.rs, builder/body_builder.rs) -- tex.1d/2d/3d, suld, sust with 5 analysis passes updated
- [x] Constant folding optimization pass (analysis/constant_folding.rs) -- simplify constant expressions at IR level
- [x] Strength reduction optimization pass (analysis/strength_reduction.rs) -- replace expensive ops with cheaper equivalents

### oxicuda-autotune (28 files, 13,039 SLoC)
- [x] Search space definition (search_space.rs) -- parameterized kernel variant spaces
- [x] Benchmark engine (benchmark.rs) -- GPU timing with warmup, statistical analysis
- [x] TunableKernel trait (tunable.rs) -- interface for autotunable kernel implementations
- [x] Configuration types (config.rs) -- tile sizes, vector widths, unroll factors
- [x] Result database (result_db.rs) -- JSON-backed per-GPU tuning result persistence
- [x] Runtime dispatcher (dispatch.rs) -- 3-tier fallback (cached > tuned > default)
- [x] Early stopping (early_stopping.rs) -- EarlyStoppingConfig, EarlyStoppingTracker, patience-based/time-budget/convergence detection
- [x] Bayesian optimization search (bayesian.rs) -- GP surrogate + acquisition functions (EI, UCB, PI)
- [x] Simulated annealing search (simulated_annealing.rs) -- temperature-based exploration for large search spaces
- [x] Genetic algorithm search (genetic.rs) -- crossover/mutation on config populations
- [x] PTX template integration (ptx_integration.rs) -- direct SearchSpace generation from template parameters
- [x] Problem size interpolation (interpolation.rs) -- nearest-neighbor and inverse-distance-weighted interpolation
- [x] Error types (error.rs) -- autotune-specific error handling

---

## Vol.3: Linear Algebra Primitives -- cuBLAS equivalent [COMPLETE]

### oxicuda-blas (72 files, 19,913 SLoC)
- [x] GpuFloat trait hierarchy (types.rs) -- F16, BF16, TF32, F32, F64, FP8
- [x] BlasHandle (handle.rs) -- session handle with stream and workspace binding
- [x] Error types (error.rs) -- BLAS-specific error variants
- [x] BLAS Level 1 -- vector-vector operations
  - [x] axpy (y = alpha * x + y)
  - [x] scal (x = alpha * x)
  - [x] dot (dot product)
  - [x] nrm2 (L2 norm)
  - [x] asum (L1 absolute sum)
  - [x] iamax (index of max absolute value)
  - [x] copy_vec (vector copy)
  - [x] swap (vector swap)
- [x] BLAS Level 2 -- matrix-vector operations
  - [x] gemv (y = alpha * A * x + beta * y)
  - [x] symv (symmetric matrix-vector multiply)
  - [x] trmv (triangular matrix-vector multiply)
  - [x] trsv (triangular solve: T * x = b)
  - [x] ger (rank-1 update: A += alpha * x * y^T)
  - [x] syr (symmetric rank-1 update)
- [x] BLAS Level 3 -- matrix-matrix operations
  - [x] gemm (general matrix multiply with dispatch)
    - [x] SIMT path (simt.rs) -- CUDA Core non-Tensor-Core path
    - [x] Tensor Core path (tensor_core.rs) -- WMMA/MMA dispatch
    - [x] Split-K parallelization (splitk.rs) -- for tall-skinny matrices
    - [x] Epilogue fusion (epilogue.rs) -- D = alpha*A@B + beta*C + bias + activation
    - [x] Dispatch logic (dispatch.rs) -- precision x arch optimal kernel selection
  - [x] gemm_api (gemm_api.rs) -- high-level GEMM entry point
  - [x] symm (symmetric matrix multiply)
  - [x] trsm (triangular solve: T * X = B)
  - [x] syrk (C = alpha * A * A^T + beta * C)
  - [x] syr2k (C = alpha * (A*B^T + B*A^T) + beta * C)
  - [x] trmm (triangular matrix multiply)
- [x] Batched GEMM operations
  - [x] batched_gemm (independent GEMM batch execution)
  - [x] strided_gemm (strided batched GEMM)
  - [x] grouped_gemm (variable-size GEMM groups)
- [x] Precision-specific optimizations
  - [x] f64_ops (FP64 DGEMM for scientific computing)
  - [x] f32_ops (FP32 SGEMM)
  - [x] f16_ops (FP16 HGEMM with Tensor Core)
  - [x] bf16_ops (BF16 GEMM with Tensor Core)
  - [x] tf32_ops (TF32 GEMM, Ampere+)
  - [x] fp8_ops (FP8 GEMM, Hopper+)
  - [x] mixed (mixed-precision accumulation)
  - [x] int_ops (INT4/INT8 GEMM for inference via dp4a-accelerated INT8 + packed INT4)
- [x] Elementwise operations
  - [x] unary (relu, gelu, sigmoid, silu, tanh, exp, log, sqrt, abs, neg)
  - [x] binary (add, sub, mul, div, max, min, pow)
  - [x] ops (fused elementwise operation dispatch)
- [x] Reduction operations
  - [x] sum (parallel sum reduction)
  - [x] max / min (parallel max/min reduction)
  - [x] mean (mean with numerically stable accumulation)
  - [x] variance (Welford online variance)
  - [x] softmax (numerically stable softmax)
  - [x] ops (reduction operation dispatch)
- [x] Complex number support (complex_gemm.rs) -- CGEMM/ZGEMM, complex_gemm, complex_gemv with interleaved storage
- [x] Batched TRSM (batched_trsm.rs) -- batched triangular solve with warp/shared/blocked strategies
- [x] Stream-K GEMM (stream_k.rs) -- dynamic work partitioning across CTAs
- [x] Persistent kernel GEMM (persistent_gemm.rs) -- work-stealing via atomic counter
  - [x] Warp-specialized GEMM (gemm/warp_specialized.rs) -- producer/consumer warps overlapping global loads with Tensor Core MMA
  - [x] Non-square tile configurations (gemm/tiles.rs) -- RectangularTile, TileSelector, aspect-ratio heuristics
  - [x] FP8 GEMM dynamic tile selection (precision/fp8_ops.rs) -- shape-dependent heuristics for Fp8WorkloadClass
  - [x] SYRK/SYR2K tensor core paths (level3/syrk_tc.rs) -- triangle-masked TC kernels
  - [x] Multi-stream batched GEMM (batched/multi_stream_batched.rs) -- distribute across multiple streams

---

## Vol.4: Deep Learning Primitives -- cuDNN equivalent [COMPLETE]

> **Known exception (2026-08-12):** two items below are checked `[ ]`/`[x]`
> per the notes inline with them, not blanket "done" -- see those lines for
> specifics. In short: Winograd forward (`conv/fprop/winograd.rs`) and
> Winograd backward (`conv/dgrad/winograd.rs`, `conv/wgrad/winograd.rs`) are
> load/launch-only skeletons -- their PTX bodies emit only step-marker
> comments and `ret`, so calling them leaves the output buffer completely
> untouched rather than numerically wrong. `conv/algo_select.rs` gates the
> forward dispatcher (`winograd_forward_implemented() == false`) so
> `conv_forward` never routes real work into the broken engine, falling
> back to `Im2colGemm`/`ImplicitGemm` instead. `conv/fused.rs`'s
> single-kernel `FusedConvBnAct` is the same kind of skeleton and is
> likewise never dispatched; the public `conv_bn_relu` decomposes into a
> real `conv_forward` call plus a real, numerically-verified BN-affine +
> activation epilogue kernel instead -- correct output, just not the
> zero-round-trip single-kernel fusion the name aspires to.

### oxicuda-dnn (89 files, 31,293 SLoC)
- [x] DnnHandle (handle.rs) -- session handle for DNN operations
- [x] Error types (error.rs) -- DNN-specific error variants
- [x] Tensor utilities (tensor_util.rs) -- layout, stride, shape helpers
- [x] DNN types (types.rs) -- tensor descriptors, data formats, algorithm enums
- [x] PTX helpers (ptx_helpers.rs) -- shared PTX generation utilities
- [x] Convolution module
  - [x] Descriptors (conv/descriptor.rs) -- ConvolutionDescriptor, FilterDescriptor
  - [x] Algorithm selection (conv/algo_select.rs) -- heuristic + autotuned selection
  - [x] High-level API (conv/api.rs) -- unified convolution interface
  - [x] Forward: implicit GEMM (conv/fprop/implicit_gemm.rs)
  - [x] Forward: im2col + GEMM (conv/fprop/im2col_gemm.rs)
  - [ ] Forward: Winograd 3x3 (conv/fprop/winograd.rs) -- load/launch-only skeleton, no numeric work; `algo_select.rs` gates it off so `conv_forward` safely falls back to Im2colGemm/ImplicitGemm instead of dispatching to it
  - [x] Forward: direct 1x1 / depthwise (conv/fprop/direct.rs)
  - [x] Backward data: implicit GEMM (conv/dgrad/implicit_gemm.rs)
  - [x] Backward filter: implicit GEMM (conv/wgrad/implicit_gemm.rs)
  - [x] Fused Conv + BN + Activation (conv/fused.rs) -- `conv_bn_relu` is decomposed (real `conv_forward` + a real combined BN-affine+activation epilogue kernel), not `FusedConvBnAct`'s single-kernel body, which remains a load/launch-only skeleton
  - [x] Transposed convolution (conv/transpose_conv.rs) -- TransposeConvConfig, TransposeConvPlan, col2im PTX, weight reshape PTX
  - [x] 3D convolution (conv/conv3d/) -- im2col3d + GEMM, forward/backward/wgrad for volumetric data
  - [x] Depthwise separable conv fusion (conv/depthwise_separable.rs) -- fused DW+PW kernel
  - [x] Deformable convolution (conv/deformable.rs) -- DCNv2 with bilinear interpolation, forward+backward
  - [x] FFT-based convolution (conv/fft_conv.rs) -- frequency-domain convolution for large kernels (7x7+)
- [x] Attention module
  - [x] Multi-Head Attention naive (attn/mha.rs)
  - [x] FlashAttention forward (attn/flash_attn/forward.rs) -- FP16, causal mask
  - [x] FlashAttention backward (attn/flash_attn/backward.rs)
  - [x] PagedAttention (attn/flash_attn/paged.rs) -- KV-cache paging
  - [x] Decode attention (attn/flash_attn/decode.rs) -- single-query inference
  - [x] Rotary Positional Embedding (attn/rope.rs) -- RoPE for LLM position encoding
  - [x] KV-Cache management (attn/kv_cache.rs)
  - [x] Fused RoPE+attention (attn/fused_rope_attn.rs) -- single kernel RoPE + attention
  - [x] FlashAttention-3 Hopper (attn/flash_attn/hopper.rs) -- warp-specialized forward+backward, TMA, ping-pong pipeline
- [x] Mixture of Experts (MoE) module
  - [x] Top-k routing (moe/routing.rs) -- softmax-based expert selection
  - [x] Token permutation (moe/permute.rs) -- scatter/gather for expert dispatch
  - [x] Fused MoE kernel (moe/fused_moe.rs) -- single-pass MoE execution
  - [x] Grouped GEMM for MoE (moe/grouped_gemm.rs)
  - [x] MoE auxiliary loss computation (moe/aux_loss.rs) -- Switch Transformer load-balance loss + z-loss
  - [x] Expert capacity factor tuning (moe/capacity.rs) -- dynamic capacity adjustment
  - [x] MoE load balancing monitoring (moe/monitoring.rs) -- runtime utilization tracking
- [x] Normalization module
  - [x] BatchNorm forward + backward (norm/batch_norm.rs)
  - [x] LayerNorm (norm/layer_norm.rs)
  - [x] RMSNorm (norm/rms_norm.rs) -- for LLM architectures (LLaMA, etc.)
  - [x] GroupNorm (norm/group_norm.rs)
  - [x] InstanceNorm (norm/instance_norm.rs) -- per (batch, channel) normalization
  - [x] ScaleNorm (norm/scale_norm.rs) -- simplified L2 normalization
  - [x] PowerNorm (norm/power_norm.rs) -- running power mean normalization
  - [x] Fused normalization (norm/fused_norm.rs) -- norm + activation fusion
- [x] Pooling module
  - [x] MaxPool2D (pool/max_pool.rs)
  - [x] AvgPool2D (pool/avg_pool.rs)
  - [x] AdaptivePool (pool/adaptive_pool.rs)
  - [x] GlobalPool (pool/global_pool.rs)
- [x] Resize module
  - [x] Nearest-neighbor interpolation (resize/nearest.rs)
  - [x] Bilinear interpolation (resize/bilinear.rs)
  - [x] Bicubic interpolation (resize/bicubic.rs)
- [x] Quantization module
  - [x] FP8 quantization (quantize/fp8_quantize.rs) -- Hopper+
  - [x] INT8 quantization (quantize/int8_quantize.rs)
  - [x] Block-scaled FP4 (quantize/block_scale.rs) -- Blackwell
- [x] RNN module
  - [x] LSTM cell (rnn/lstm.rs) -- full forward pass for single timestep and sequence
  - [x] GRU cell (rnn/gru.rs) -- full forward pass for single timestep and sequence
- [x] INT4/NF4 quantization (quantize/int4_quantize.rs) -- quantize/dequantize with group scaling (QLoRA support)
- [x] GQA/MQA native support (attn/gqa.rs) -- grouped-query and multi-query attention
- [x] Sliding window attention (attn/sliding_window.rs) -- Mistral-style configurable window size
- [x] Fused GEMM+bias+activation epilogue (fused_linear.rs) -- fused_linear with all activations
  - [ ] Winograd backward pass (conv/dgrad/winograd.rs, conv/wgrad/winograd.rs) -- same load/launch-only skeleton pattern as the forward pass (no numeric work); `conv_backward_data`/`conv_backward_filter` do not dispatch to it (they use the separate implicit-GEMM dgrad/wgrad engines), so it is only exercised directly by gpu_tests canaries
  - [x] Block-sparse attention (attn/block_sparse.rs) -- CSR-format structured sparsity patterns for long-context
  - [x] Quantization-aware training (quantize/qat.rs) -- fake quantize + straight-through estimator

---

## Vol.5.5: GPU Parallel Primitives -- CUB Equivalent [COMPLETE]

### oxicuda-primitives (16 files, ~4,200 SLoC)

CUB-equivalent parallel GPU primitives, zero CUDA SDK dependency.
All kernels are generated as PTX source strings at runtime and JIT-compiled via `cuModuleLoadData`.

- [x] Error types (error.rs) -- PrimitivesError with 9 variants, PrimitivesResult alias
- [x] Session handle (handle.rs) -- PrimitivesHandle wrapping Arc<Context> + Arc<Stream> + SmVersion
- [x] Shared PTX utilities (ptx_helpers.rs) -- ptx_header, ptx_type_str/bytes, reg_decl, ReduceOp, PrimitiveType trait
  - Full SmVersion coverage including Sm90a, Sm100, Sm120
  - Full PtxType coverage including BF16x2, F16x2, TF32, FP8 (E4M3/E5M2), FP6, FP4 (E2M1), B128
- [x] **Warp-level primitives** (warp/reduce.rs, warp/scan.rs)
  - Reduce: `shfl.sync.bfly.b32` butterfly tree; f64 split lo/hi; optional broadcast lane
  - Scan: `shfl.sync.up.b32` shift-and-combine; inclusive + exclusive; f64 lo/hi split
  - All 7 ops: Sum, Product, Min, Max, And, Or, Xor
- [x] **Block-level primitives** (block/reduce.rs, block/scan.rs)
  - Reduce: warp-reduce + shared-memory merge + `shfl.sync.bfly` for warp-level aggregation
  - Scan: work-efficient Blelloch up-sweep / down-sweep; inclusive + exclusive
  - All 7 ops; f64 split-register support throughout
- [x] **Device-wide reduce** (device/reduce.rs) -- 2-pass pipeline (partial sums → final scalar)
- [x] **Device-wide scan** (device/scan.rs) -- 3-kernel pipeline: block scan → propagate → apply
- [x] **Stream compaction** (device/select.rs) -- 2-kernel flag+gather pipeline over exclusive scan
  - `SelectPredicate`: NonZero, Positive, Negative (unsigned → always false), FlagArray
  - Type-correct `setp.{lt,gt,ne}.{ty}` per predicate × element type
- [x] **Privatized histogram** (device/histogram.rs) -- 2-kernel init+count pipeline
  - `DeviceHistogramMode`: Modulo (rem.u32/rem.u64) and EvenRange (fp or integer linear map)
  - Per-block shared-memory private histogram; `atom.shared.add.u32`; strided global merge
- [x] **4-bit LSD Radix Sort** (sort/radix_sort.rs) -- 3 kernels per pass × 8 (u32) or 16 (u64) passes
  - Count kernel: private `cnt_hist[16]` in shared memory + `atom.shared.add.u32`
  - Scan kernel: 1 block × 16 threads; sequential column scan for exclusive prefix
  - Scatter kernel: `block_offs[16]` pre-loaded + `atom.shared.add.u32` for unique output positions
- [x] **Bitonic Block Sort + Co-rank Merge Sort** (sort/merge_sort.rs)
  - Bitonic sort: 2-barrier-per-stage correctness (pre-load + pre-write); `selp.{ty}` for type-correct compare-swap
  - Merge kernel: O(log n) co-rank binary search per element; branch-based output selection
  - 154 tests: 142 unit + 12 doctests, all passing
  - ptx_helpers comprehensive coverage: all PtxType variants (B128, E4M3/E5M2/E2M3/E3M2/E2M1, F16x2, BF16x2, TF32), all SmVersion variants (Sm90a, Sm100, Sm120), all ReduceOp identities and instruction mnemonics

---

## Vol.5: Scientific Computing & Ecosystem Integration [COMPLETE]

### oxicuda-fft (35 files, 8,853 SLoC)
- [x] FftPlan (plan.rs) -- transform planning with strategy selection
- [x] FFT execution engine (execute.rs) -- plan dispatch and execution
- [x] Error types (error.rs) -- FFT-specific errors
- [x] FFT types (types.rs) -- Complex<f32>, Complex<f64>, transform direction
- [x] PTX helpers (ptx_helpers.rs) -- FFT kernel PTX generation utilities
- [x] Stockham auto-sort FFT (kernels/stockham.rs) -- GPU-optimized in-place FFT
- [x] Batched FFT (kernels/batch_fft.rs) -- multiple small FFTs in parallel
- [x] Large FFT (kernels/large_fft.rs) -- global memory multi-pass for large sizes
- [x] Matrix transpose (kernels/transpose.rs) -- for multi-dimensional FFT decomposition
- [x] Radix-2 butterfly (radix/radix2.rs)
- [x] Radix-4 butterfly (radix/radix4.rs)
- [x] Radix-8 butterfly (radix/radix8.rs)
- [x] Mixed-radix support (radix/mixed_radix.rs) -- composite sizes (2, 3, 5, 7)
- [x] Bluestein / Chirp-Z (radix/bluestein.rs) -- arbitrary-size FFT
- [x] Complex-to-Complex (transforms/c2c.rs)
- [x] Real-to-Complex (transforms/r2c.rs)
- [x] Complex-to-Real (transforms/c2r.rs)
- [x] 2D FFT (transforms/fft2d.rs)
- [x] 3D FFT (transforms/fft3d.rs)
- [x] Stockham bank conflict avoidance (bank_conflict_free.rs) -- padded layout (addr + addr/32), power-of-2 sizes 64-4096
- [x] Batched FFT kernel fusion (fused_batch.rs) -- multiple small FFTs per thread block, N<=1024, shared memory only
- [x] Split-radix FFT (split_radix.rs) -- radix-2/4 hybrid for ~10% fewer operations
- [x] Real-valued FFT optimization (real_fft.rs) -- pack/unpack exploiting conjugate symmetry
- [x] Prime-factor algorithm (pfa.rs) -- Good-Thomas FFT with CRT mapping
- [x] Convolution via FFT helper (conv_fft.rs) -- 1D/2D convolution + cross-correlation
  - [x] Multi-GPU FFT (multi_gpu.rs) -- 1D slab decomposition across P devices

### oxicuda-sparse (36 files, 11,021 SLoC)
- [x] Sparse handle (handle.rs)
- [x] Error types (error.rs)
- [x] PTX helpers (ptx_helpers.rs) -- sparse kernel PTX utilities
- [x] Storage formats
  - [x] CSR -- Compressed Sparse Row (format/csr.rs)
  - [x] CSC -- Compressed Sparse Column (format/csc.rs)
  - [x] COO -- Coordinate format (format/coo.rs)
  - [x] BSR -- Block Sparse Row (format/bsr.rs)
  - [x] ELL -- ELLPACK format (format/ell.rs)
  - [x] Format conversion (format/convert.rs) -- CSR<->CSC, COO<->CSR, etc.
- [x] Sparse operations
  - [x] SpMV -- sparse matrix-vector multiply (ops/spmv.rs)
  - [x] SpMM -- sparse matrix-matrix multiply (ops/spmm.rs)
  - [x] SpGEMM -- sparse-sparse matrix multiply (ops/spgemm.rs)
  - [x] SDDMM -- sampled dense-dense matrix multiply (ops/sddmm.rs)
  - [x] SpTRSV -- sparse triangular solve (ops/sptrsv.rs)
  - [x] Krylov subspace methods (ops/krylov.rs) -- Lanczos and Arnoldi iteration
- [x] Preconditioners
  - [x] ILU(0) -- incomplete LU factorization (preconditioner/ilu0.rs)
  - [x] IC(0) -- incomplete Cholesky factorization (preconditioner/ic0.rs)
- [x] ELL-optimized SpMV (spmv_ell.rs) -- coalesced column-major access, sentinel-based padding
- [x] BSR SpMV kernel (spmv_bsr.rs) -- block-aware SpMV, one thread block per block-row, dense sub-block multiply
- [x] CSR5 format and SpMV (csr5.rs, spmv_csr5.rs) -- tile-based CSR variant with Csr5Matrix, two-phase SpMV (tile + calibration)
- [x] Graph coloring for parallel ILU/IC (graph_coloring.rs) -- distance-2 greedy coloring, parallel_ilu0
- [x] Multi-level ILU(k) (preconditioner/iluk.rs) -- symbolic + numeric with configurable fill levels
- [x] Sparse matrix reordering (reorder.rs) -- RCM and AMD ordering
- [x] Merge-based SpGEMM (ops/spgemm_merge.rs) -- load-balanced with merge-path
- [x] SpMV auto-format selection (ops/auto_spmv.rs) -- heuristic CSR/ELL/BSR/CSR5 selection
- [x] SpGEMM memory estimation (ops/spgemm_estimate.rs) -- upper bound, exact, sampling strategies

### oxicuda-solver (40 files, 13,981 SLoC)
- [x] Solver handle (handle.rs)
- [x] Error types (error.rs)
- [x] PTX helpers (ptx_helpers.rs)
- [x] Dense solvers
  - [x] LU factorization (dense/lu.rs)
  - [x] QR factorization (dense/qr.rs)
  - [x] SVD -- Singular Value Decomposition (dense/svd.rs)
  - [x] Cholesky factorization (dense/cholesky.rs)
  - [x] Eigenvalue decomposition (dense/eig.rs)
  - [x] Matrix inverse (dense/inverse.rs)
  - [x] Determinant computation (dense/det.rs)
  - [x] Least squares (dense/lstsq.rs)
  - [x] Matrix functions (dense/matrix_functions.rs) -- expm, logm, sqrtm via Pade approximation
- [x] Sparse / iterative solvers
  - [x] Conjugate Gradient (sparse/cg.rs)
  - [x] BiCGSTAB (sparse/bicgstab.rs)
  - [x] GMRES (sparse/gmres.rs)
  - [x] Direct sparse solver (sparse/direct.rs)
- [x] Helper utilities
  - [x] Condition number estimation (helpers/condition.rs)
  - [x] Pivoting strategies (helpers/pivot.rs)
- [x] Batched LU/QR/Cholesky (batched.rs) -- BatchedSolver for many small matrices (4x4 to 64x64), batched_solve
- [x] Randomized SVD (randomized_svd.rs) -- Halko-Martinsson-Tropp 2011, configurable rank/oversampling/power iterations
- [x] Preconditioned CG/GMRES (preconditioned.rs) -- Preconditioner trait, Jacobi, PCG, PGMRES
- [x] Tridiagonal/pentadiagonal solvers (tridiagonal.rs) -- Thomas algorithm, cyclic reduction, batched
- [x] Divide-and-conquer SVD (dense/dc_svd.rs) -- recursive bidiagonal splitting
- [x] Symmetric indefinite factorization (dense/ldlt.rs) -- Bunch-Kaufman LDL^T
- [x] Flexible GMRES (sparse/fgmres.rs) -- variable preconditioner per iteration
- [x] Band matrix solvers (dense/band.rs) -- banded LU, Cholesky, solve

### oxicuda-rand (27 files, 9,064 SLoC)
- [x] RNG generator handle (generator.rs) -- unified RNG interface
- [x] Error types (error.rs)
- [x] RNG engines
  - [x] Philox 4x32-10 counter-based RNG (engines/philox.rs)
  - [x] MRG32k3a combined multiple recursive (engines/mrg32k3a.rs) -- with matrix power skip-ahead for parallel MC
  - [x] XORWOW (engines/xorwow.rs)
- [x] Distributions
  - [x] Uniform (distributions/uniform.rs)
  - [x] Normal / Gaussian (distributions/normal.rs)
  - [x] Log-Normal (distributions/log_normal.rs)
  - [x] Poisson (distributions/poisson.rs)
- [x] Quasi-random sequences
  - [x] Sobol sequences (quasi/sobol.rs)
- [x] Philox kernel optimization (philox_optimized.rs) -- 4 values per thread, grid-stride loop, Box-Muller pair generation
- [x] Scrambled Sobol sequences (scrambled_sobol.rs) -- Owen's scrambling for improved equidistribution
- [x] Binomial distribution (distributions/binomial.rs) -- direct inversion + BTPE algorithm
- [x] Geometric distribution (distributions/geometric.rs) -- inverse CDF method
- [x] Halton sequences (quasi/halton.rs) -- multi-dimensional quasi-random
- [x] Latin Hypercube sampling (quasi/latin_hypercube.rs) -- stratified space-filling design
- [x] Multinomial distribution (distributions/multinomial.rs) -- conditional-binomial decomposition
- [x] Truncated normal (distributions/truncated_normal.rs) -- accept-reject Box-Muller

### oxicuda (umbrella crate) (44 files, 18,764 SLoC)
- [x] Re-exports all sub-crates under unified namespace
- [x] ComputeBackend trait (backend.rs) -- ComputeBackend trait, CudaBackend implementation, feature-gated for SciRS2 integration; alloc/free/copy_htod/copy_dtoh/synchronize/init now use real oxicuda_driver calls (PrimaryContext + cuMemAlloc_v2/cuMemcpyHtoD_v2/cuMemcpyDtoH_v2/cuCtxSynchronize); reduce/unary/binary pending PTX pipeline wiring
  - [x] Global initialization (global_init.rs) -- OxiCudaRuntime singleton with device auto-selection
  - [x] OxiONNX GPU inference backend (onnx_backend/) -- IR graph, op implementations, executor, planner, fusion, shape inference
  - [x] ToRSh GPU backend (tensor_backend/) -- tensor, dtype, autograd, ops, optimizer, mixed precision
  - [x] TrustformeRS Transformer GPU backend (transformer_backend/) -- KV-cache, attention, scheduler, speculative decoding, sampling, quantization

---

## Vol.6: Signal Processing, Audio & Image Primitives [COMPLETE]

### oxicuda-signal (13 files, ~3,500 SLoC, 231 tests)

GPU-accelerated signal processing: DCT, DWT, MDCT, STFT, window functions, FIR/IIR filters, image processing — all with CPU reference implementations and PTX kernel generators.

- [x] **Core scaffolding**
  - [x] Error types (error.rs) -- SignalError with 6 variants, SignalResult alias
  - [x] Handle (handle.rs) -- SignalHandle with SmVersion + stream
  - [x] Types (types.rs) -- WaveletFamily, WindowType, SignalPrecision, PadMode
  - [x] PTX helpers (ptx_helpers.rs) -- ptx_header, global_tid_1d, bounds_check, next_pow2
- [x] **DCT transforms** (dct/)
  - [x] DCT-II CPU reference + twiddle PTX kernel (dct2.rs)
  - [x] DCT-III CPU reference + pre-twiddle/un-permute PTX kernels (dct3.rs)
  - [x] DCT-IV CPU reference + PTX pre/post twiddle (dct4.rs)
  - [x] MDCT / IMDCT + sine window + KBD window + MdctPlan (mdct.rs)
- [x] **DWT wavelets** (dwt/)
  - [x] Haar forward/inverse CPU + PTX kernels (haar.rs)
  - [x] Daubechies db2–db10 forward/inverse CPU (daubechies.rs) -- filter tables, conv_downsample
  - [x] Symlet sym2–sym10 forward (sym.rs)
  - [x] Multi-level DWT: forward/inverse + WaveletDecomposition, soft/hard/universal threshold (multilevel.rs)
- [x] **Audio processing** (audio/)
  - [x] STFT / windowed DFT + StftConfig + magnitude/power spectrogram (stft.rs)
  - [x] Window functions: Hann, Hamming, Blackman, Blackman-Harris, Kaiser, Bartlett, Gaussian, FlatTop, Dolph-Chebyshev (stft.rs)
  - [x] Mel filterbank + MFCC + delta/delta-delta coefficients (mel.rs)
  - [x] Audio spectrogram metrics: SNR, peak, LUFS, spectral centroid/rolloff/flatness, MFCC distance (spectrogram.rs)
- [x] **Window analysis** (window.rs)
  - [x] Coherent gain, ENBW, process gain, peak sidelobe level
  - [x] PTX window-apply kernel (element-wise multiply)
  - [x] Standard window catalog
- [x] **FIR/IIR filters** (filter/)
  - [x] FIR design: lowpass, highpass, bandpass, bandstop (windowed sinc) + raised cosine / RRC (fir.rs)
  - [x] FIR application: direct-form with zero/circular/reflect/replicate padding + freq response
  - [x] FIR PTX kernel (direct-form short filter ≤ 64 taps)
  - [x] IIR Biquad sections: lowpass, highpass, bandpass, peaking EQ + freq response (iir.rs)
  - [x] General-order IIR apply (Direct Form II Transposed)
  - [x] Butterworth pole design + SOS cascade
  - [x] Wiener filter: spectral estimation + gain computation + batch apply (wiener.rs)
- [x] **Correlation** (correlation/)
  - [x] Cross-correlation, autocorrelation, normalized correlation coefficient (crosscorr.rs)
  - [x] Phase correlation (sub-pixel peak estimation) for image alignment
  - [x] GCC-PHAT (Generalized Cross-Correlation with Phase Transform)
- [x] **Image processing** (image/)
  - [x] Separable Gaussian blur: 1D kernel generation + H/V pass + full 2D blur + PTX kernels (gaussian_blur.rs)
  - [x] Sobel edge detection: Gx/Gy/magnitude/angle + PTX kernels (sobel.rs)
  - [x] Morphological operations: dilate/erode/open/close/top-hat/gradient + structuring elements (morphology.rs)
  - [x] Non-Maximum Suppression: bounding box IoU + greedy NMS + soft-NMS + heatmap NMS (nms.rs)

---

## Vol.7: Computation Graph Engine [COMPLETE]

### oxicuda-graph (11 files, ~4,800 SLoC, 175 tests)

High-level DAG-based computation graph engine that sits above the raw CUDA driver.
Models GPU workloads as computation graphs, applies analysis and optimisation passes,
and lowers to `oxicuda_driver::graph::Graph` for low-overhead CUDA graph submission.

- [x] **Core scaffolding**
  - [x] Error types (error.rs) -- GraphError (10 variants), GraphResult alias
  - [x] Node types (node.rs) -- NodeId, BufferId, StreamId, KernelConfig, MemcpyDir, NodeKind (8 variants), GraphNode, BufferDescriptor
  - [x] ComputeGraph DAG (graph.rs) -- adjacency-list DAG, eager cycle detection, topo sort (Kahn), reachability, DOT export
  - [x] GraphBuilder API (builder.rs) -- fluent builder with auto data-flow edge inference from buffer I/O annotations
- [x] **Analysis passes** (analysis/)
  - [x] Topological analysis (topo.rs) -- ASAP/ALAP scheduling, slack, critical path, level assignment, priority ordering
  - [x] Liveness analysis (liveness.rs) -- buffer live intervals [def_pos, last_use_pos] in topo order, interference pairs, peak live bytes
  - [x] Dominance analysis (dominance.rs) -- Cooper et al. dominator tree, idom, dominates(), dominated_by(), LCA
- [x] **Optimisation passes** (optimizer/)
  - [x] Operator fusion (fusion.rs) -- greedy chain fusion of compatible element-wise kernels using dominator + config checks
  - [x] Memory planning (memory.rs) -- live-interval graph colouring (best-fit), 256-byte alignment, pool layout
  - [x] Stream partitioning (stream.rs) -- list-scheduling heuristic, predecessor-aware stream assignment, cross-stream sync detection
- [x] **Executor backends** (executor/)
  - [x] ExecutionPlan (plan.rs) -- full compilation pipeline (topo→liveness→fusion→memory→stream→linearise), PlanStep sequence with event record/wait pairs
  - [x] Sequential executor (sequential.rs) -- CPU-side simulation, event validity checking, ExecutionStats
  - [x] CUDA graph capture (cuda_graph.rs) -- converts ExecutionPlan to oxicuda_driver::graph::Graph with dependency edges

---

## Vol.8: GPU Training Engine [COMPLETE]

### oxicuda-train (15 files, ~4,200 SLoC, 105 tests)

Production-grade GPU-accelerated training utilities implementing the v1.2 roadmap items:
gradient checkpointing, mixed-precision optimizer states, and large-scale distributed training.

- [x] **Error types and session handle** (error.rs, handle.rs)
  - [x] `TrainError` (12 variants) with `TrainResult<T>` alias
  - [x] `TrainHandle` — wraps `Arc<Context>` + `Arc<Stream>` + SM version metadata; `device_sm_version()` via CUdevice_attribute
- [x] **PTX update kernels** (ptx_kernels.rs)
  - [x] `adam_update_ptx` — fused moment update + bias-corrected Adam step; `fma.rn.f32`, `sqrt.approx.f32`, `rcp.approx.f32`
  - [x] `adamw_update_ptx` — decoupled weight decay: `p *= (1 − lr·wd)` before moment update
  - [x] `sgd_update_ptx` — Nesterov SGD with `setp.ne.f32` predicate for conditionality
  - [x] `lion_update_ptx` — sign via bit-mask: `and.b32 sign_bit, c_bits, 0x80000000`
  - [x] `came_row_factor_ptx` / `came_col_factor_ptx` — per-row/col accumulation for CAME factored second moment
  - [x] `norm_sq_partial_ptx` — block-level ‖g‖² with warp butterfly `shfl.sync.bfly.b32` + smem merge
  - [x] `scale_inplace_ptx` / `add_inplace_ptx` — element-wise scale and gradient accumulation
  - [x] All kernels: grid-stride `$LOOP`/`$DONE`, sm_80/sm_90 PTX header selection, `f32_hex()` IEEE literals
- [x] **GPU optimizers** (gpu_optimizer/)
  - [x] `GpuAdam` — bias-corrected first+second moments, optional AMSGrad variant
  - [x] `GpuAdamW` — decoupled weight decay (default wd=0.01); differs from Adam's L2 regularization
  - [x] `GpuLion` — single moment buffer; sign update `p = p·(1−lr·λ) − lr·sign(c)`; 50% memory vs Adam
  - [x] `GpuCame` — factored second moment: `CameV::Matrix { row, col }` O(m+n) vs O(mn) for Adam
  - [x] `GpuMuon` — Nesterov + Newton-Schulz orthogonalisation; 5-iteration `X ← 1.5X − 0.5X·XᵀX`
  - [x] `GpuOptimizer` trait: `step()`, `zero_grad()`, `lr()`, `set_lr()`, `name()`
  - [x] `adam_bias_corrections()` helper: pre-computes `step_size = lr/(1−β₁ᵗ)` and `bc2_rsqrt = 1/√(1−β₂ᵗ)`
- [x] **Gradient utilities** (grad_clip.rs, grad_accum.rs)
  - [x] `GlobalNormClip` — joint ‖g‖ across all params; f64 accumulation; scale = max_norm/(norm+ε)
  - [x] `PerLayerClip` / `ValueClip` — independent per-param and element-wise clipping
  - [x] `GradientAccumulator` — k micro-batch accumulation; `Average` and `Sum` reduction modes
- [x] **Gradient checkpointing** (checkpoint.rs)
  - [x] `CheckpointPolicy`: Uniform { interval }, Selective { names }, Offload, None
  - [x] `CheckpointManager` — save/retrieve/recompute activation segments; `RecomputeFn` closures
  - [x] `CheckpointOverflow` error when segment count exceeds `max_segments`
- [x] **LR Schedulers** (lr_scheduler.rs) — 11 variants via `LrScheduler` trait
  - [x] ConstantLR, StepLR, MultiStepLR, ExponentialLR (with `base_lr()` getters)
  - [x] CosineAnnealingLR, LinearWarmup, WarmupCosine, PolynomialDecayLR, OneCycleLR, CyclicLR
  - [x] ReduceLROnPlateau — metric-based reduction with patience and min_lr floor
- [x] **ZeRO distributed optimizer** (zero.rs)
  - [x] `ZeroStage`: Stage1/2/3 with `shard_range(n) = (rank*chunk, min(start+chunk, n))`
  - [x] `ZeroOptimizer<O: GpuOptimizer>` — wraps any optimizer; Stage2 zeros non-owned gradients; Stage3 operates only on owned parameter shard
  - [x] `ZeroMemoryEstimate` — `bytes_per_rank()` and `reduction_ratio()` capacity planning helpers
- [x] **Integration tests** (lib.rs) — 6 E2E tests: AdamW+WarmupCosine+clip, Lion+accumulation, CAME+CyclicLR, Muon+ReduceLROnPlateau, ZeRO-2, checkpoint+recompute

---

## Vol.9: GPU-Accelerated Reinforcement Learning [COMPLETE]

### oxicuda-rl (29 files, ~6,090 SLoC, 165 tests)

First-class GPU-ready RL library implementing every major modern algorithm from DQN to SAC/TD3/PPO.

- [x] **Error types and session handle** (error.rs, handle.rs)
  - [x] `RlError` (12 variants): DimensionMismatch, InsufficientTransitions, InvalidPriority, InvalidConfig, EmptyBatch, NanEncountered, InvalidLogProb, NanLoss, EpisodeError, InvalidStateSize, InvalidAction, Other
  - [x] `SmVersion(u32)` with `ptx_version_str()` mapping sm≥100→"8.7", sm≥90→"8.4", sm≥80→"8.0", else "7.5"
  - [x] `LcgRng` — 64-bit LCG (multiplier 6364136223846793005) with `next_u32()`, `next_f32()`, `next_usize(n)`
  - [x] `RlHandle::default_handle()` — sm=80, device=0, seed=42
- [x] **PTX kernel sources** (ptx_kernels.rs) — 5 GPU kernels
  - [x] `td_error_ptx` — TD-error `δ = r + γ*(1-done)*V' - V`, grid-stride
  - [x] `normalize_advantages_ptx` — mean/variance normalisation pass
  - [x] `ppo_ratio_ptx` — clipped importance ratio `exp(lp_new - lp_old)` with `ex2.approx.f32`
  - [x] `sac_target_ptx` — soft Bellman target `y = r + γ*(1-done)*(min(Q1,Q2) - α*lp)`
  - [x] `per_is_weight_ptx` — IS weight `(N*p_i)^{-β}` normalised by max; `lg2.approx.f32`
- [x] **Experience replay buffers** (buffer/)
  - [x] `UniformReplayBuffer` — struct-of-arrays circular buffer; rejection sampling without replacement
  - [x] `PrioritizedReplayBuffer` — dual sum+min segment tree O(log N); stratified sampling across strata; IS weight computation
  - [x] `NStepBuffer` — circular buffer of `Option<Step>`; n-step return accumulation with γ^n bootstrap; flush on episode end
- [x] **Policy distributions** (policy/)
  - [x] `CategoricalPolicy` — Gumbel-max sampling; log-prob; entropy; KL-divergence; greedy; log_prob_batch
  - [x] `GaussianPolicy` — Box-Muller N(0,1); reparameterisation μ+σ⊙ε; Tanh squashing with Jacobian correction; log-prob batch
  - [x] `DeterministicPolicy` — DDPG exploration noise; TD3 target policy smoothing (clipped Gaussian); `OrnsteinUhlenbeck` OU process
- [x] **Return / advantage estimators** (estimator/)
  - [x] `compute_gae` — backward scan `A_t = δ_t + γλ(1-done)A_{t+1}`; optional Welford normalisation; GAE-λ
  - [x] `compute_td_lambda` — `G_t = r_t + γ*mask*[(1-λ)*v_{t+1} + λ*G_{t+1}]`; takes values[T+1] bootstrap
  - [x] `compute_vtrace` — IMPALA V-trace: c_t=min(c̄,ρ_t), ρ̄_t=min(ρ̄,ρ_t); backward scan advantages
  - [x] `compute_retrace` — safe off-policy Q-targets: c_t=λ*min(1,ρ_t); `Q^ret_t = Q_t + δ_t + γ*c_{t+1}*(Q^ret_{t+1}-Q_{t+1})`
- [x] **RL algorithm loss functions** (loss/)
  - [x] `ppo_loss` — clip ratio*A + `PpoConfig{clip_eps=0.2, c_v=0.5, c_e=0.01}`; approx_kl, clip_fraction metrics
  - [x] `dqn_loss` / `double_dqn_loss` — Bellman MSE/Huber (kappa=1.0); IS-weighted; Double-DQN decoupled selection
  - [x] `sac_critic_loss` / `sac_actor_loss` / `sac_temperature_loss` — entropy-regularized; log-space temperature
  - [x] `td3_critic_loss` / `td3_actor_loss` — twin-Q Bellman error + deterministc actor `-mean(Q1_μ)`
- [x] **Normalization** (normalize/)
  - [x] `RunningStats` — Welford online N-dim: δ=x-mean; mean+=δ/n; M2+=δ*δ₂; dim() accessor; batch update
  - [x] `ObservationNormalizer` — wraps RunningStats; clip=5.0; enable/disable; eval mode (no stat update)
  - [x] `RewardNormalizer` — `ReturnNorm` (G_t=γG_{t-1}+r_t, divide by std), `Clip`, `None` modes; n_envs parallel
- [x] **Environment abstractions** (env/)
  - [x] `Env` trait — `obs_dim()`, `act_dim()`, `reset()`, `step()`, `is_continuous()`
  - [x] `LinearQuadraticEnv` — s'=0.9s+a+noise, r=-s²-0.1a²; Box-Muller noise (>> 41 bit shift, NaN-safe)
  - [x] `VecEnv<E: Env>` — batched `reset_all()`, `step()` (auto-reset on done), `foreach()`, `terminal_obs` tracking
- [x] **Integration tests** (lib.rs) — 5 E2E tests
  - [x] `e2e_dqn_style_loop` — collect 200 transitions + DQN loss on LQ env
  - [x] `e2e_ppo_gae_loss` — 128-step GAE + PPO clip+value+entropy loss
  - [x] `e2e_sac_style_update` — PER buffer + SAC critic loss with IS weights
  - [x] `e2e_vecenv_with_obs_norm` — 4×VecEnv 20 steps + ObservationNormalizer Welford update
  - [x] `e2e_n_step_buffer` — 3-step return verification: R ≈ 1+0.99+0.99²

---

## Vol.10: Quantization & Model Compression Engine [COMPLETE]

### oxicuda-quant (24 files, ~5,442 SLoC, 151 tests)

Post-training quantization (PTQ), quantization-aware training (QAT), pruning,
knowledge distillation, and mixed-precision analysis for LLM and DNN deployment.

- [x] **Error types** (error.rs) — 12 `QuantError` variants
  - DimensionMismatch, EmptyInput, InvalidScale, InvalidBitWidth, GroupSizeMismatch, CalibrationRequired, SingularHessian, TeacherStudentMismatch, AllZeroPruning, NonFiniteFp8, InfeasibleCompressionTarget, InvalidConfig
- [x] **PTX kernels** (ptx_kernels.rs) — 5 GPU-side quantization kernels
  - `fake_quant_ptx` — STE-aware fake quantization
  - `int8_quant_ptx` / `int8_dequant_ptx` — INT8 quant/dequant with scale+zp
  - `nf4_dequant_ptx` — NF4 lookup table in shared memory
  - `prune_mask_ptx` — apply sparsity mask in-place
- [x] **Quantization schemes** (scheme/)
  - [x] `MinMaxQuantizer` (minmax.rs) — INT4/INT8 Symmetric/Asymmetric PerTensor/PerChannel/PerGroup; 9 tests
  - [x] `Nf4Quantizer` (nf4.rs) — QLoRA NF4 with exact quantile LUT, nibble packing, absmax blocks; 8 tests
  - [x] `Fp8Codec` (fp8.rs) — E4M3 (max=448) and E5M2 (max=57344) via IEEE 754 bit manipulation; 8 tests
  - [x] `GptqQuantizer` (gptq.rs) — Hessian OBC via Cholesky + L⁻¹, column-wise weight correction; 8 tests
  - [x] `SmoothQuantMigrator` (smooth_quant.rs) — α-scaled activation/weight migration, preserves output; 8 tests
- [x] **QAT observers and fake quant** (qat/)
  - [x] `MinMaxObserver` — running global min/max, compute scale/zp; 5 tests
  - [x] `MovingAvgObserver` — EMA momentum update of min/max; 3 tests
  - [x] `HistogramObserver` — histogram + min-MSE percentile clipping search; 3 tests
  - [x] `FakeQuantize` — quantize→dequantize forward, STE backward; enabled/disabled mode; 10 tests
- [x] **Pruning** (pruning/)
  - [x] `SparseMask` — boolean weight mask; sparsity(), apply(), apply_in_place(), and/or compose; 7 tests
  - [x] `MagnitudePruner` — L1/L2 unstructured with grouped variant; 8 tests
  - [x] `StructuredPruner` — channel/filter/head granularity, L2-norm unit ranking; 6 tests
- [x] **Knowledge distillation** (distill/)
  - [x] `DistilLoss` — KL(τ²-scaled), MSE, cosine, combined; 10 tests
  - [x] `ResponseDistiller` — soft+hard label combination, batch loss; 5 tests
  - [x] `FeatureDistiller` — per-layer weighted feature matching, normalise_weights; 7 tests
- [x] **Compression analysis** (analysis/)
  - [x] `SensitivityAnalyzer` — per-layer MSE across bit-widths via MinMax symmetric; 8 tests
  - [x] `CompressionMetrics` + `ModelCompressionMetrics` — bits, ratio, sparsity, weighted MSE; 7 tests
  - [x] `MixedPrecisionPolicy` — greedy sensitivity-guided bit assignment; 7 tests

---

## Vol.11: High-Performance Inference Engine [COMPLETE]

### oxicuda-infer (22 files, ~5,900 SLoC, 138 tests)

vLLM-style continuous batching inference engine with PagedAttention KV cache,
speculative decoding, beam search, and a pluggable ModelRunner abstraction.

- [x] **Error types and session handle** (error.rs, handle.rs)
  - [x] `InferError` (15 variants): BlockAllocFailed, InvalidSequenceId, EmptyBatch, DimensionMismatch, SamplingError, SchedulerFull, NoPrefillSeqs, CacheManagerError, InvalidSamplingParams, EosTokenMissing, BeamSearchError, SpeculativeError, UnsupportedConfig, ModelRunnerError, Other
  - [x] `InferHandle` — device, sm_version, n_layers, n_heads, n_kv_heads, head_dim, vocab_size, block_size, max_seq_len; `ptx_version_str()`, `attention_scale()`

- [x] **PTX kernel sources** (ptx_kernels.rs) — 5 GPU-side inference kernels
  - [x] `paged_attn_ptx` — online Flash-Attention-style softmax over paged KV blocks; per-block numerically-stable `m_new = max(m, tile_max)`
  - [x] `rope_apply_ptx` — in-place RoPE with `cos.approx.f32` / `sin.approx.f32`; frequency `θ_i = position * 10000^{-2i/d}`
  - [x] `top_k_filter_ptx` — sets non-top-K logit positions to NEG_INFINITY; register-shuffle warp sort
  - [x] `logits_softmax_ptx` — three-pass stable softmax: max→sum_exp→normalize using warp butterfly reduces
  - [x] `kv_append_ptx` — writes K/V into physical block slot; grid-stride across attention heads

- [x] **KV cache** (cache/)
  - [x] `BlockId(u32)` opaque identifier; `KvBlock` with `append()`, `key_slice()`, `value_slice()`, `reset()`
  - [x] `PagedKvCache` — `[n_layers][n_blocks]` 2D block pool; O(1) free-list alloc; reference counting for copy-on-write prefix sharing; 8 tests
  - [x] `CacheManager` — per-sequence block tables `HashMap<u64, Vec<BlockId>>`; auto-grow on block fill; `allocate_sequence`, `free_sequence`, `append_token`; 7 tests
  - [x] `PrefixCache` — FNV-1a token hash → `PrefixEntry`; LRU eviction; `lookup()`, `insert()`, `hit_rate()`; 9 tests

- [x] **Batch scheduling** (batch/)
  - [x] `SequenceStatus`: Waiting → Prefill → Decode → Finished(FinishReason) with EosToken(u32) / MaxLength variants
  - [x] `SamplingParams` — temperature, top_k, top_p, max_new_tokens, eos_token_id, repetition_penalty; 8 tests
  - [x] `Scheduler` — FCFS admission; token-budget decode phase; memory-pressure preemption; `ScheduledBatch{prefill_ids, decode_ids}`; `on_step_complete` / `take_finished`; 9 tests
  - [x] `ContinuousBatcher` — orchestrates scheduler + cache_manager + model_fn + Rng; one batched forward pass per `step()`; 8 tests

- [x] **Sampling suite** (sampling/)
  - [x] `Rng` — 64-bit LCG (Knuth constants); `next_u64()`, `next_f32()`, `next_usize(n)`; `softmax` + `categorical_sample`; 5 tests
  - [x] `greedy_sample` / `greedy_sample_batch` — argmax with NaN guard; 7 tests
  - [x] `top_k_filter` / `top_k_sample` — threshold from k-th sorted logit; exactly-k tokens retained; 7 tests
  - [x] `top_p_filter` / `top_p_sample` — sorted cumulative-probability nucleus cutoff; 5 tests
  - [x] `BeamSearchState::step()` — log-softmax expansion; keep beam_width candidates; EOS → completed; length-normalised `score/len^α`; 8 tests
  - [x] `speculative_verify()` — rejection sampling: accept `d_i` if `u < min(1, p_target/p_draft)`; correction token from `max(0,p−q)/Z`; provably identical distribution to target; 6 tests

- [x] **Executor** (executor/)
  - [x] `ModelRunner` trait — `vocab_size()`, `decode(token_ids, block_tables, seq_lens)`, `prefill(token_ids, seq_starts, block_tables)`; 9 tests
  - [x] `MockModelRunner` — peaks at `(token_id + bias) % vocab_size`; deterministic for unit testing
  - [x] `RunnerStats` — n_steps, total_tokens, sequences_completed; `avg_batch_size()`
  - [x] `paged_attention_cpu` — reference GQA PagedAttention: load K/V per block, Q·K^T·scale, stable softmax, weighted ×V; kv_h=h/(n_heads/n_kv_heads); `AttentionConfig`; 5 tests

- [x] **Integration tests** (lib.rs) — 6 E2E tests
  - [x] `e2e_greedy_until_eos` — continuous batching generates until EOS token
  - [x] `e2e_max_tokens_termination` — max_new_tokens=1 path
  - [x] `e2e_beam_search_completes` — beam_width=2 finishes on EOS in one step
  - [x] `e2e_speculative_all_accepted` — draft==target → all k drafts accepted
  - [x] `e2e_paged_attention_single_token` — Q=V=1.0 → output=1.0
  - [x] `e2e_prefix_cache_hit_rate` — 3 queries (1 miss, 2 hits) → hit_rate=2/3

---

## Vol.12: Distributed Inference Engine [COMPLETE]

### oxicuda-dist-infer (20 files, ~5,800 SLoC, 80 tests)

Multi-GPU distributed inference with three orthogonal parallelism axes (TP × SP × EP = world_size),
distributed KV-cache management, and affinity-aware request routing.

- [x] **Error types and session handle** (error.rs, handle.rs)
  - [x] `DistInferError` (27 variants): InvalidWorldSize, RankOutOfRange, TooFewRanks, TpFeaturesMisaligned, TpInputMisaligned, ShardShapeMismatch, SpSeqLenMisaligned, EmptyChunk, EpExpertsMisaligned, EmptyExpertBatch, SequenceNotOwned, MigrationTargetInvalid, BlockPoolExhausted, AllRanksAtCapacity, EmptyTokenSequence, NoPrefixAffinity, DimensionMismatch, Internal
  - [x] `ParallelismConfig { tp, sp, ep }` — three-way parallelism decomposition, `world_size()`, `validate()`
  - [x] `RankCoordinates` — 3-D tp/sp/ep coordinates from flat global rank; `peer_tp/sp/ep()` for ring lookups
  - [x] `DistInferHandle` — lightweight descriptor with device, SM version, config, coords; `single_rank()` for tests

- [x] **PTX kernel sources** (ptx_kernels.rs) — 5 GPU-side kernels
  - [x] `tp_col_scatter_ptx` — column-parallel linear scatter: write strided shard into full output buffer
  - [x] `tp_row_all_reduce_ptx` — row-parallel linear all-reduce: ring partial-sum accumulation
  - [x] `sp_seq_chunk_copy_ptx` — sequence chunk copy: extract/insert contiguous token slice (direction=0/1)
  - [x] `ep_token_scatter_ptx` — expert-parallel token scatter: route tokens to expert-local input buffers
  - [x] `ep_token_gather_ptx` — expert-parallel token gather: collect expert outputs back to original order

- [x] **Tensor parallelism** (tensor_parallel/)
  - [x] `ColumnLinearShard` — weight shard `[local_out × in]`; `forward()` local GEMM; `validate()`
  - [x] `ColumnLinear` — column-parallel linear: `from_full_weight()` slices rows; `local_forward()`; `all_gather()` simulates collective
  - [x] `RowLinearShard` — weight shard `[out × local_in]`; `forward_partial()` local GEMM; bias only on rank 0
  - [x] `RowLinear` — row-parallel linear: `from_full_weight()` slices columns; `slice_input()`; `all_reduce()` simulates ring reduce

- [x] **Sequence parallelism** (sequence_parallel/)
  - [x] `ChunkInfo` — describes rank's token window: start, len, total_tokens, hidden_dim
  - [x] `SeqSplitter` — `extract_chunk()`, `insert_chunk()`, `all_gather()`, `reduce_scatter()`; validates divisibility
  - [x] `BoundaryExchange` — pre-attention all-gather of K/V; post-attention reduce-scatter of outputs; `local_attention()` with causal masking and GQA-compatible head indexing

- [x] **Expert parallelism** (expert_parallel/)
  - [x] `TopKRouter` — top-K selection from gating logits + softmax weight normalisation; `RoutingPlan` with expert_load; `load_balance_cv()` metric
  - [x] `RoutingEntry` / `RoutingPlan` — per-(token,expert) assignment with routing weight
  - [x] `LocalExpertBatch` — dispatched token batch per expert with token_indices and weights
  - [x] `ExpertDispatcher` — `scatter()` → local expert buffers; `gather()` → weighted output sum; `dispatch_and_gather()` end-to-end

- [x] **Distributed KV cache** (distributed_cache/)
  - [x] `SeqOwnership` / `RankCacheStats` — per-sequence owner rank + block count; per-rank utilization stats
  - [x] `CachePartition` — least-loaded assignment; `grow()`, `release()`; `rebalance_suggestions()` (utilization-threshold migration hints); `apply_migration()`
  - [x] `BlockData` — serialized KV block `[n_layers × 2 × block_size × kv_dim]`; `key_slice(l)` / `value_slice(l)`; `validate()`
  - [x] `MigrationRequest` / `MigrationStats` — cross-rank block transfer descriptor + statistics
  - [x] `BlockMigrator` — `receive_block()` → local staging id; `take_block()`; `validate_target()`; stats tracking

- [x] **Request routing** (router/)
  - [x] `Request` — token_ids, max_new_tokens, priority; `prefix_hash(len)` FNV-1a for affinity lookup
  - [x] `RoutingDecision` / `DispatchPolicy` — selected rank + policy tag + prefix_hit flag
  - [x] `RankLoad` — free_blocks, total_blocks, in_flight; `utilization()`
  - [x] `RouterMetrics` — per-policy request counts, total_routed, prefix_hits, `prefix_hit_rate()`
  - [x] `RoutingPolicy` — three modes: RoundRobin, LeastLoaded, PrefixAffinity (with fallback + registration)

- [x] **Integration tests** (lib.rs) — 6 E2E tests
  - [x] `e2e_tp_column_row_roundtrip` — tp=4 column-parallel + all-gather + row-parallel + all-reduce = identity
  - [x] `e2e_sp_attention_pipeline` — sp=2 extract chunks + all-gather + local_attention (uniform QKV → output=1.0)
  - [x] `e2e_ep_moe_dispatch_gather` — ep=2, 4 experts, 4 tokens, top-1 routing + identity experts + gather
  - [x] `e2e_cache_partition_lifecycle` — 4 ranks, 8 sequences, assign/grow/release lifecycle
  - [x] `e2e_routing_prefix_affinity_pipeline` — first request misses, second with same prefix hits same rank
  - [x] `e2e_ptx_kernels_all_sm_versions` — all 5 kernels × 5 SM versions produce valid PTX headers

---

## Vol.13: LLM Inference Primitives [COMPLETE]

### oxicuda-lm (16 files, ~3,200 SLoC, 182 tests)

Model-layer abstractions for LLM inference: BPE tokenizer, transformer layer building blocks
with KV-cache for incremental decode, complete GPT-2 and LLaMA-2/3 model implementations,
and GPU kernel PTX string generators.

- [x] **Error types and config** (error.rs, config.rs)
  - [x] `LmError` (17 variants): DimensionMismatch, InvalidConfig, EmptyInput, OutOfVocab, Utf8Decode, WeightNotFound/ShapeMismatch, LayerIndexOutOfRange, HeadDimMismatch, KvCacheLengthMismatch, SequenceTooLong, InvalidMergePair, VocabSizeMismatch, GqaHeadMismatch, WeightDataLengthMismatch, Internal
  - [x] `GptConfig` — GPT-2 presets: `gpt2_small` (12L/12H/768D), `gpt2_medium` (24L/16H/1024D), `gpt2_large`, `gpt2_xl`, `tiny` (2L/2H/8D)
  - [x] `LlamaConfig` — LLaMA presets: `llama2_7b`, `llama2_13b`, `llama3_8b` (GQA 32H/8KV), `mistral_7b`, `phi2`, `tiny` (2L/4H/2KV)
  - [x] `SmVersion` + `LmHandle` (handle.rs) — SM version with `ptx_version_str()`, `target_str()`

- [x] **Weights** (weights.rs)
  - [x] `WeightTensor { data, shape }` — `zeros()`, `ones()`, `eye()`, `from_data()`, `row_slice()`, `validate_shape()`
  - [x] `ModelWeights` (HashMap-backed) — `get_checked()` with shape validation, `n_params()`, iterators

- [x] **PTX kernel generators** (ptx_kernels.rs) — 5 GPU kernels × 6 SM versions (75/80/86/90/100/120)
  - [x] `embedding_forward_ptx` — token embedding table lookup (grid-stride over n_tokens×embed_dim)
  - [x] `rope_apply_ptx` — RoPE in-place from pre-computed cos/sin tables; grid-stride pair indexing
  - [x] `silu_gate_ptx` — SwiGLU gate: `out = (g/(1+exp(-g))) * up`; `ex2.approx.f32` + `rcp.approx.f32`
  - [x] `rms_norm_ptx` — shared-memory warp butterfly reduction → normalize + scale; `sqrt.approx.f32`
  - [x] `causal_attn_softmax_ptx` — per-head causal mask + stable softmax (max → exp → sum → normalize)

- [x] **Tokenizer** (tokenizer/)
  - [x] `Vocab` (vocab.rs) — byte↔id bidirectional map; `gpt2_byte_vocab()` (256 byte tokens); `with_extra_tokens()`, `special_id()`
  - [x] `BpeTokenizer` (bpe.rs) — byte-level BPE; `merge_ranks` (priority table) + `pair_to_merged` (result table); `encode()` vocab-lookup init → greedy lowest-rank merge loop; `decode()` byte concat → UTF-8
  - [x] `BpeBuilder` — `add_merge()`, `add_special()`, `build()` convenience builder

- [x] **Layers** (layer/)
  - [x] `RmsNorm` / `LayerNorm` (norm.rs) — per-token normalize with learnable weight (and bias for LayerNorm)
  - [x] `TokenEmbedding`, `LearnedPositionalEmbedding`, `RotaryEmbedding` (embedding.rs) — RoPE with precomputed cos/sin tables, absolute position offset for KV-cache decode
  - [x] `MlpFfn` (ffn.rs) — GPT-2 GELU MLP: `W_proj(GELU(W_fc·x+b))+b_proj`
  - [x] `SwiGluFfn` (ffn.rs) — LLaMA SwiGLU: `W_down(silu(W_gate·x) ⊙ W_up·x)`, no biases
  - [x] `LayerKvCache` / `MultiHeadAttention` (attention.rs) — GQA (`kv_h = q_h / (n_heads/n_kv_heads)`), causal mask at absolute position `past_len + t`, KV append for incremental decode
  - [x] `GptBlock` / `LlamaBlock` / `PastKvCache` (transformer.rs) — pre-LN residual blocks; multi-layer KV cache container

- [x] **Models** (model/)
  - [x] `Gpt2Model` (model/gpt.rs) — token+pos embedding → N×GptBlock → LayerNorm → weight-tied LM head; `next_token()` greedy decode
  - [x] `LlamaModel` (model/llama.rs) — TokenEmbedding → N×LlamaBlock → RmsNorm → independent LM head; `next_token()` greedy decode
  - [x] Weight loaders (model/weights.rs) — `load_gpt2_block()` (HuggingFace key convention, packed QKV split), `load_llama_block()` (separate q/k/v proj)

- [x] **Integration tests** (lib.rs) — 10 E2E tests
  - [x] GPT-2 tiny forward (shape, zero-weight → zero-logits)
  - [x] LLaMA tiny forward (shape validation)
  - [x] GPT-2 incremental decode consistency (full vs token-by-token last-position logit match)
  - [x] LLaMA incremental decode consistency
  - [x] BPE encode/decode round-trip ("hello" → [259] → "hello")
  - [x] RMSNorm + LayerNorm numerical correctness
  - [x] PTX kernels × 6 SM versions (target directive presence)
  - [x] LLaMA GQA multi-step decode (prefill 4 + decode 3 → past_len=7)
  - [x] Vocab special token round-trip (BOS/EOS)
  - [x] GPT-2 greedy decode loop (5 steps, all IDs in vocab range)

## Vol.14–16: SDE Samplers in oxicuda-rand [COMPLETE]

Added stochastic differential equation (SDE) integration methods as a new `sde` submodule
in `oxicuda-rand`, providing GPU-simulation building blocks for financial models, physical
simulations, and score-based generative model training.

### oxicuda-rand SDE additions (4 files, ~1,060 SLoC, 71 new tests)

- [x] **SDE framework** (sde/mod.rs) — shared `SdeProcess` trait, `SdeConfig`, `PathMatrix`, Xoshiro256** PRNG
- [x] **Brownian motion** (sde/brownian.rs) — `BrownianMotion`, `GeometricBrownianMotion` (exact), `OrnsteinUhlenbeck` (exact), `BrownianPathResult` with covariance check
- [x] **Euler-Maruyama** (sde/euler_maruyama.rs) — strong order 0.5 scheme for `dX = μ dt + σ dW`; `EulerMaruyamaResult` with mean/std/path statistics; `strong_error()` comparison
- [x] **Milstein** (sde/milstein.rs) — strong order 1.0 via `½σσ'(ΔW² − Δt)` correction; `convergence_comparison()` verifying EM vs Milstein accuracy on GBM
- [x] **Stratonovich-Heun** (sde/heun.rs) — predictor-corrector for Stratonovich SDEs; `solve_ito()` with automatic Itô→Stratonovich correction `μ_strat = μ − ½σσ'`

---

## Vol.17: Generative AI Primitives [COMPLETE]

### oxicuda-gen (25 files, ~7,400 SLoC, 221 tests)

Pure-Rust generative AI primitives: diffusion schedulers (DDPM/DDIM/DPM-Solver++/Flow Matching),
classifier-free guidance, VQ-VAE codec, LoRA adapters, and score-network building blocks.

- [x] **Error types** (error.rs) — `GenError` (15 variants): DimensionMismatch, InvalidBetaRange, InvalidGuidanceScale, UnsupportedDpmOrder, InvalidTimestep, InvalidCodebookSize, WeightShapeMismatch, InvalidLoraRank, and more
- [x] **Handle** (handle.rs) — `SmVersion`, `LcgRng` (seed-based, Box-Muller normals), `GenHandle`

- [x] **PTX kernels** (ptx_kernels.rs) — 6 GPU kernels × 6 SM versions (75/80/86/90/100/120)
  - [x] `ddpm_step_ptx` — `x_{t-1} = (x_t − β/√(1−ᾱ) · ε̂)/√α + σ·z` using `sqrt.approx`, `rcp.approx`
  - [x] `cfg_combine_ptx` — `out = u + s·(c − u)` classifier-free guidance blend
  - [x] `lora_apply_ptx` — `y = x·W + (α/r)·x·B·A` low-rank update; grid-stride
  - [x] `flow_velocity_ptx` — Euler step `x_{t+Δ} = x_t + Δ·v(x_t, t)` for flow ODE
  - [x] `vae_kl_loss_ptx` — `0.5·Σ(μ² + σ² − 1 − log σ²)` latent KL divergence
  - [x] `timestep_embed_ptx` — sinusoidal embedding via `sin/cos/lg2/ex2`

- [x] **Schedulers** (scheduler/) — 5 files
  - [x] `BetaSchedule` (beta_schedule.rs) — linear, cosine (Nichol & Dhariwal), scaled-cosine, sigmoid β schedules; `alphas_bar`, `sqrt_alphas_bar`, `sqrt_one_minus_alphas_bar`
  - [x] `DdpmScheduler` (ddpm.rs) — `add_noise()` with `q(xₜ|x₀)`, `step()` reverse DDPM update with fixed σ²=βₜ
  - [x] `DdimScheduler` (ddim.rs) — η-parameterised deterministic/stochastic; η=0 → identical two-call results
  - [x] `DpmSolverScheduler` (dpm_solver.rs) — exponential integrator on `λₜ = log(αₜ/σₜ)`; 1st/2nd-order multi-step; `num_train_steps()` accessor
  - [x] `FlowMatchingScheduler` (flow_matching.rs) — linear OT path `xₜ = (1−t)x₀+tx₁`; Euler and Heun ODE solvers; boundary conditions verified in tests

- [x] **Guidance** (guidance/) — 3 files
  - [x] `CfgGuidance` (cfg.rs) — `ε̂ = uncond + s·(cond − uncond)` with scale-clipping and rescaling
  - [x] `PerpNegGuidance` (perp_neg.rs) — perpendicular-negative prompt guidance
  - [x] `AdaptiveCfgScheduler` (adaptive.rs) — constant/linear/cosine/stepwise dynamic scale scheduling

- [x] **VAE** (vae/) — 4 files
  - [x] `GaussianLatent` (kl.rs) — reparameterised sampling `z = μ + ε·σ`, `kl_loss()`, `standard_normal()`
  - [x] `VqCodebook` (quantize.rs) — EMA codebook update `eₖ ← γeₖ + (1−γ)∑xⱼ`, nearest-entry lookup, commitment loss
  - [x] `Encoder` (encoder.rs) — ResNet down-blocks (GELU + GroupNorm); `EncoderWeights::zeros()`
  - [x] `Decoder` (decoder.rs) — mirrored up-sampling blocks; `DecoderWeights::zeros()`

- [x] **LoRA** (lora/) — 2 files
  - [x] `LoraLinear` (adapter.rs) — `W' = W + (α/r)·BA`; B∈ℝᵈˣʳ Gaussian init, A=0 init; `forward()` adds rank-r correction
  - [x] `LoraModel` (adapter.rs) — named adapter collection with `add_adapter()`/`apply()`
  - [x] Weight merging (merge.rs) — `merge_lora()`, `unmerge_lora()`, `verify_merge_roundtrip()`, `scale_adapter()`, `compose_adapters()`

- [x] **Score networks** (score/) — 2 files
  - [x] `SinusoidalEmbedding` / `FourierEmbedding` (timestep.rs) — sin+cos pair embedding with sin²+cos²=1 verified
  - [x] `UNetResBlock`, `SelfAttentionBlock`, `CrossAttentionBlock` (unet_block.rs) — SiLU + time-embedding injection + multi-head attention

- [x] **Integration tests** (lib.rs) — 11 E2E tests covering all modules + PTX × 6 SM versions

---

## Vol.18: Graph Neural Network Primitives [COMPLETE]

### oxicuda-gnn (25 files, ~7,370 SLoC, 233 tests)

Pure-Rust GNN library: sparse graph representations (CSR/COO/heterogeneous), message passing,
GCN/GAT/GATv2/GraphSAGE/GIN layers, global and hierarchical pooling, Set2Set readout.

- [x] **Error types** (error.rs) — `GnnError` (14 variants): EmptyGraph, NodeIndexOutOfRange, EdgeIndexOutOfRange, InvalidLayerConfig, FeatureDimensionMismatch, InvalidEdgeWeight, InvalidPoolingK, SamplingError, and more

- [x] **Handle** (handle.rs) — `SmVersion`, `GnnHandle`, `LcgRng`

- [x] **PTX kernels** (ptx_kernels.rs) — 7 GPU kernels × 6 SM versions
  - [x] `csr_spmv_ptx` — `y[i] = Σ A[i,j]·x[j]`; warp-per-row with `shfl.sync.down` butterfly reduction
  - [x] `scatter_add_ptx` — `out[idx[i]] += in[i]`; `atom.global.add.f32`
  - [x] `gat_attention_ptx` — `LeakyReLU(aᵀ[Wxᵢ‖Wxⱼ])` per edge
  - [x] `softmax_edge_ptx` — numerically stable per-source softmax over outgoing edges
  - [x] `aggregate_mean_ptx` — accumulator / `degree[i]` mean reduction
  - [x] `gin_combine_ptx` — `(1+ε)·xᵢ + Σxⱼ` self-loop aggregator
  - [x] `topk_score_ptx` — `tanh(pᵀx/‖p‖)` scoring for Top-K node selection

- [x] **Graph representations** (graph/) — 4 files
  - [x] `CsrGraph` (csr.rs) — `row_ptr/col_idx/edge_weight`; `from_edges()`, `neighbors()`, `degrees()`, `normalized_adjacency()` (D̂⁻¹/²ÂD̂⁻¹/²)
  - [x] `CooGraph` (coo.rs) — COO format with `to_csr()` conversion; symmetry detection
  - [x] `HeterogeneousGraph` (heterogeneous.rs) — multi-type node/edge relations
  - [x] `KHopSubgraph`, random walk, Node2Vec biased walk (sampling.rs)

- [x] **Message passing** (message_passing/) — 3 files
  - [x] Aggregations (aggregate.rs) — sum/mean/max/min/softmax over neighbor messages
  - [x] Scatter ops (scatter.rs) — `scatter_add`, `scatter_max`, `scatter_min`, `scatter_mul`, `scatter_softmax`
  - [x] Update functions (update.rs) — MLP (2-layer), identity, ReLU, SiLU, LeakyReLU

- [x] **GNN Layers** (layers/) — 5 files
  - [x] `GcnLayer` (gcn.rs) — `H⁽ˡ⁺¹⁾ = σ(D̂⁻¹/²ÂD̂⁻¹/² H⁽ˡ⁾ W⁽ˡ⁾)` (Kipf & Welling 2017)
  - [x] `GatLayer` (gat.rs) — `αᵢⱼ = softmax(LeakyReLU(aᵀ[Wxᵢ‖Wxⱼ]))`, multi-head concat/mean (Veličković 2018)
  - [x] `GatV2Layer` (gat_v2.rs) — dynamic attention `aᵀLeakyReLU(W[xᵢ‖xⱼ])` (Brody 2022)
  - [x] `SageLayer` (sage.rs) — mean/MaxPool/LSTM aggregators; optional L2-norm output (Hamilton 2017)
  - [x] `GinLayer` (gin.rs) — `(1+ε)·hᵥ + Σhᵤ` with MLP; BatchNorm; trainable ε (Xu 2019)

- [x] **Pooling** (pooling/) — 3 files
  - [x] `GlobalPool` (global_pool.rs) — mean/max/sum/attention pooling to graph-level repr; batched graphs
  - [x] `TopKPool` (topk_pool.rs) — Gao & Ji top-k node selection with `tanh(pᵀx/‖p‖)` scoring
  - [x] `DiffPool` (diff_pool.rs) — differentiable hierarchical: `S=softmax(GNN(A,X))`, `X'=SᵀX, A'=SᵀAS`; LP + entropy regularisation losses

- [x] **Readout** (readout/) — 1 file
  - [x] `Set2Set` (set2set.rs) — LSTM-based permutation-invariant readout: `qₜ=LSTM(q*_{t-1})`, `αᵢₜ=softmax(xᵢᵀqₜ)`, `q*ₜ=[qₜ‖rₜ]` (Vinyals 2016)

- [x] **Integration tests** (lib.rs) — 12 E2E tests covering CSR, COO, scatter, GCN, SAGE, GIN, DiffPool, Top-K, sampling, Set2Set, PTX × 6 SM versions

---

## Vol.19: State Space Model Primitives [COMPLETE]

### oxicuda-mamba (25 files, ~7,800 SLoC, 339 tests)

Pure-Rust SSM library: S4 (HiPPO-LegS / DPLR), Mamba selective scan (S6), Mamba-2 (SSD),
and RWKV time-mixing — linear-time alternatives to attention, zero CUDA SDK dependency.

- [x] **Error types** (error.rs) — `MambaError` (15 variants): DimensionMismatch, ShapeMismatch, EmptyInput, InvalidSeqLen, InvalidSsmOrder, InvalidModelDim, NonPositiveDelta, InvalidChunkSize, HeadDimMismatch, WeightShapeMismatch, NonFinite, Internal
- [x] **Handle** (handle.rs) — `SmVersion`, `LcgRng` (Box-Muller normals, Fisher-Yates shuffle), `MambaHandle`

- [x] **PTX kernels** (ptx_kernels.rs) — 7 GPU kernels × 6 SM versions (75/80/86/90/100/120)
  - [x] `selective_scan_ptx` — Mamba S6: `h = Ā·h + B̄·u, y = C·h` per-channel sequential recurrence
  - [x] `parallel_scan_ptx` — Warp-level `(A,b)` associative prefix scan via `shfl.sync.down.b32` butterfly
  - [x] `depthwise_conv1d_ptx` — Causal 1D depthwise conv with zero-pad, `fma.rn.f32`
  - [x] `wkv_forward_ptx` — RWKV WKV with numerically stable running-max pivot; `ex2.approx.f32`
  - [x] `ssd_chunk_ptx` — Mamba-2 SSD chunk: causal `Π A_k` accumulation per output position
  - [x] `hippo_legendre_ptx` — HiPPO-LegS forward Euler `c_n' = c_n·(1−Δ(n+1)) + Δ√(2n+1)·u`
  - [x] `rms_norm_silu_ptx` — Fused RMSNorm + SiLU gate; warp butterfly sum via `shfl.sync.bfly.b32`

- [x] **SSM core** (ssm/) — 3 files
  - [x] `discretize.rs` — ZOH (`Ā = exp(Δ·A)`), Bilinear (Tustin), Euler; L'Hôpital limit for `|A| ≈ 0`
  - [x] `parallel_scan.rs` — `ScanPair {a,b}` with associative `⊕` operator, inclusive/exclusive prefix scan, `ssm_state_scan(a_bar, b_bar_u)`
  - [x] `ssm_kernel.rs` — `SsmKernel`: batch-aware `h[b,t,d,n] = Ā·h_prev + B̄·u` recurrence, ZOH discretization, output `y = Σ C·h`

- [x] **S4 architecture** (s4/) — 3 files
  - [x] `hippo.rs` — `hippo_legs(n)`: HiPPO-LegS A matrix (lower-triangular, `A[n,k]=−√(2n+1)√(2k+1)`) and B vector; `hippo_legs_diag`; `hippo_nplr` NPLR decomposition (`λ[n]=−(n+0.5)`, `p=q=√(n+0.5)`)
  - [x] `dplr.rs` — `Dplr {lambda, p, q}`: `A = diag(λ) − p·qᵀ`; `from_hippo`, `to_dense`, ZOH SSM kernel computation via mode decomposition
  - [x] `s4_layer.rs` — `S4Layer`: multi-channel convolutional mode, `naive_conv1d` O(L²) reference, optional bidirectional averaging, `S4Config` builder

- [x] **Mamba** (mamba/) — 3 files
  - [x] `selective_scan.rs` — `selective_scan`: input-dependent `Δ=softplus(proj)`, `Ā=exp(Δ⊗A)`, `B̄=Δ⊗B_proj`, sequential state recurrence; `softplus` with ±20 stability clamp
  - [x] `mamba_block.rs` — `MambaBlock`: in_proj → x/z split → conv1d+SiLU → selective_scan → D skip → SiLU gate → out_proj + residual; `rms_norm`, `linear`, `silu`, `causal_depthwise_conv1d` helpers
  - [x] `mamba_model.rs` — `MambaModel`: TokenEmbedding → N×MambaBlock → RMSNorm → LM head; `forward` returns logits, `next_token` greedy decode; `MambaConfig::tiny()` test preset

- [x] **Mamba-2 / SSD** (mamba2/) — 3 files
  - [x] `ssd.rs` — `ssd_naive` O(L²·N) semi-separable matrix-vector product; `ssd_recurrent` O(L·N) state form; `verify_ssd_equivalence` agreement check (tol 1e-4)
  - [x] `chunk_scan.rs` — `ChunkConfig` with ceiling-division chunks; `chunk_scan`: intra-chunk naive SSD + inter-chunk boundary state propagation; `verify_chunk_equivalence`
  - [x] `mamba2_block.rs` — `Mamba2Block`: multi-head SSD with `a[t]=sigmoid(−exp(a_h))`, per-head `chunk_scan`, D skip, RMSNorm, out_proj + residual

- [x] **RWKV** (rwkv/) — 3 files
  - [x] `time_mixing.rs` — `WkvState {a,b,p}` recurrent state; numerically stable WKV via running-max pivot; `TimeMixingLayer` full RWKV-4 pipeline: LN → token-shift → r/k/v projection → WKV → sigmoid gate → output projection; `layer_norm`, `sigmoid` helpers
  - [x] `channel_mixing.rs` — `ChannelMixingLayer`: token-shift → sigmoid-gated receptance → Square-ReLU expansion → value contraction; `square_relu(x) = max(0,x)²`
  - [x] `rwkv_block.rs` — `RwkvBlock`: pre-norm residual: `y = x + time_mixing(LN₁(x))`, `y = y + channel_mixing(LN₂(y))`

- [x] **Integration tests** (lib.rs) — 20 E2E tests covering all modules + PTX × 6 SM versions

---

---

## Vol.20: Vision Transformer & CLIP Primitives [COMPLETE]

### oxicuda-vision (25 files, ~7,500 SLoC, 349 tests)

Pure-Rust vision library: ViT patch embedding, multi-head self-attention, CLIP
contrastive learning, image augmentation, FPN multi-scale features, DETR decoder,
and bipartite set matching — zero CUDA SDK dependency.

- [x] **Error types** (error.rs) — `VisionError` (15 variants): DimensionMismatch, ShapeMismatch, EmptyInput, InvalidImageSize, InvalidPatchSize, InvalidEmbedDim, InvalidNumHeads, HeadDimMismatch, InvalidNumClasses, InvalidProjDim, NonPositiveTemperature, InvalidRoiBox, WeightShapeMismatch, NonFinite, Internal

- [x] **Handle** (handle.rs) — `SmVersion`, `LcgRng` (Box-Muller normals, Fisher-Yates shuffle), `VisionHandle`

- [x] **PTX kernels** (ptx_kernels.rs) — 7 GPU kernels × 6 SM versions (75/80/86/90/100/120)
  - [x] `patch_embed_ptx` — Strided Conv2D: `[C, H, W] → [N_patches, embed_dim]` with `fma.rn.f32`
  - [x] `bilinear_interp_ptx` — Sub-pixel 4-tap bilinear sampler with half-pixel convention
  - [x] `contrastive_loss_ptx` — InfoNCE: 3-pass numerically stable row-softmax + diagonal CE
  - [x] `roi_align_ptx` — Per-bin bilinear RoI feature extraction with `sampling_ratio²` sample averaging
  - [x] `image_normalize_ptx` — Channel-wise `(x − mean[c]) / std[c]` in-place
  - [x] `adaptive_avg_pool_ptx` — Adaptive 2D average pool with integer window bounds
  - [x] `focal_loss_ptx` — Focal loss `−α(1−p)^γ log p` via stable sigmoid + log

- [x] **Patch embedding** (patch_embed/) — 2 files
  - [x] `conv2d_patch.rs` — `PatchEmbedConfig`, `PatchEmbedWeights` (Xavier init), `PatchEmbed::forward`, `prepend_cls`
  - [x] `pos_embed.rs` — `pos_2d_sincos` (4-band H/W sinusoidal), `LearnablePosEmbed`, `add_pos_embed`

- [x] **ViT** (vit/) — 3 files
  - [x] `vit_block.rs` — `ViTBlock`: pre-norm MHSA + GELU MLP + residuals; `layer_norm`, `gelu_exact` (tanh approx), `softmax_rows`, `mhsa`
  - [x] `vit_encoder.rs` — `ViTEncoder`: N stacked `ViTBlock` + final LayerNorm
  - [x] `vit_model.rs` — `ViTModel`: PatchEmbed → CLS-prepend → PosEmbed → Encoder → head; `ViTConfig::tiny()` (img=32, p=4, D=64, depth=2, heads=4, classes=10)

- [x] **CLIP** (clip/) — 3 files
  - [x] `vision_encoder.rs` — `ClipVisionEncoder` wrapping `ViTEncoder`, CLS-pool to `[embed_dim]`
  - [x] `projection.rs` — `ProjectionHead`: linear + L2-norm; `cosine_sim`
  - [x] `contrastive.rs` — `info_nce_loss`: symmetric InfoNCE with numerically stable log-sum-exp

- [x] **Augmentation** (augment/) — 3 files
  - [x] `geometric.rs` — `random_crop`, `center_crop`, `random_horizontal_flip`, `resize_bilinear` (half-pixel bilinear)
  - [x] `photometric.rs` — `color_jitter` (brightness/contrast/saturation), `random_grayscale` (YIQ luminance)
  - [x] `normalize.rs` — `normalize_chw`, `IMAGENET_MEAN`/`IMAGENET_STD`; `AugOp` enum + `Pipeline::push` builder

- [x] **FPN** (fpn/) — 2 files
  - [x] `lateral.rs` — `LateralConv1x1`: 1×1 conv channel reduction (Xavier init)
  - [x] `top_down.rs` — `Fpn`: lateral → top-down (nearest upsample + add) → 3×3 smooth conv; `FeatureMap {data, channels, h, w}`

- [x] **Detection** (detection/) — 3 files
  - [x] `roi_align.rs` — CPU reference RoI Align with `bilinear_sample_2d`; validates `x2>x1, y2>y1`
  - [x] `detr_decoder.rs` — `DetrDecoderLayer`: self-attn + cross-attn + FFN (pre-norm); `DetrDecoder` depth stack; `DetrConfig::tiny()`
  - [x] `set_match.rs` — `bipartite_match` (greedy + 2-opt); `build_cost_matrix` (class CE + L1 box + GIoU); `giou`

- [x] **Integration tests** (lib.rs) — 19 E2E tests covering all modules + PTX × 6 SM versions

---

## Vol.21: Audio/Speech ML Architectures [COMPLETE]

### oxicuda-audio (28 files, ~7,500 SLoC, 286 tests)

Pure-Rust audio/speech ML library: Conformer encoder, Wav2Vec2 CNN feature extractor,
CTC forward algorithm + prefix beam search, WaveNet dilated stack, SpecAugment
augmentation, speaker embeddings (x-vector TDNN, attentive pooling) — zero CUDA SDK
dependency.

- [x] **Error types** (error.rs) — `AudioError` (17 variants): DimensionMismatch, ShapeMismatch, EmptyInput, InvalidNumMels, InvalidSequenceLength, InvalidEmbedDim, InvalidNumHeads, HeadDimMismatch, InvalidVocabSize, InvalidBeamWidth, InvalidDilation, InvalidKernelSize, InvalidStride, BlankOutOfRange, WeightShapeMismatch, NonFinite, Internal

- [x] **Handle** (handle.rs) — `SmVersion`, `LcgRng` (Box-Muller normals, Fisher-Yates shuffle), `AudioHandle::default_handle()` (SM 8.0, device 0, seed 42)

- [x] **PTX kernels** (ptx_kernels.rs) — 7 GPU kernels × 6 SM versions (75/80/86/90/100/120)
  - [x] `stride_conv1d_ptx` — Strided 1-D conv for Wav2Vec2 CNN feature extractor
  - [x] `dilated_conv1d_ptx` — Causal dilated conv (WaveNet filter+gate, left-pad)
  - [x] `ctc_alpha_ptx` — Log-domain CTC forward alpha recursion with `log_sum_exp` via `ex2`/`lg2`
  - [x] `spec_augment_mask_ptx` — In-place time+freq masking via `setp`/`selp.f32`
  - [x] `depthwise_conv1d_ptx` — Causal depthwise conv for Conformer conv module
  - [x] `rel_pos_bias_ptx` — Relative-position bias table lookup with `min/max.u32` clamping
  - [x] `stats_pool_ptx` — Two-pass mean+std pooling with warp-shuffle reduction

- [x] **Features** (features/) — 3 files
  - [x] `log_mel_adapter.rs` — `LogMelInput` validated `[T, F]` wrapper for `oxicuda-signal` output
  - [x] `cmvn.rs` — `CmvnConfig`, `compute_cmvn`, `apply_cmvn` (per-channel zero-mean unit-variance)
  - [x] `delta.rs` — `compute_delta`, `compute_delta_delta`, `stack_delta_features` (central-difference, edge-pad)

- [x] **Encoder** (encoder/) — 3 files
  - [x] `wav2vec_cnn.rs` — `Wav2VecCnnEncoder`: 7-layer stride-conv1d + group-norm + GELU; `wav2vec2_base()` and `tiny()` configs
  - [x] `conv_module.rs` — `ConvModule`: LN → PW-expand → GLU → depthwise-causal → BN → Swish → PW-reduce
  - [x] `conformer_block.rs` — `ConformerBlock` (macaron ½·FFN + MHSA(rel-pos) + ConvModule + ½·FFN + LN) + `ConformerEncoder`; `ConformerConfig::tiny()` (D=64, heads=4, depth=2, kernel=15)

- [x] **Attention** (attention/) — 2 files
  - [x] `rel_pos_encoding.rs` — `RelPosEncoding {table: [2*max_len-1]}` with seeded init, `bias(q,k)`, `bias_matrix(Q,K)`
  - [x] `rel_pos_attention.rs` — `RelPosAttention`: multi-head SDPA + relative-position bias pre-softmax

- [x] **CTC** (ctc/) — 2 files
  - [x] `forward.rs` — `ctc_forward_log`: log-domain alpha recursion, extended target `l'=[blank,l0,blank,l1,…]`, `log_sum_exp2` stable
  - [x] `beam_search.rs` — `ctc_beam_search`: CTC prefix beam search with `HashMap<Vec<usize>, (p_blank, p_nb)>` merge and pruning

- [x] **Vocoder** (vocoder/) — 2 files
  - [x] `wavenet_block.rs` — `WaveNetBlock`: dilated-causal-conv → tanh⊙sigmoid gated activation → skip + residual pointwise convs
  - [x] `dilated_stack.rs` — `WaveNetStack`: multi-cycle `[1,2,4,…,512]` dilation schedule + 2-layer ReLU head; `tiny()` and `default_config()`

- [x] **Augmentation** (augment/) — 2 files
  - [x] `spec_augment.rs` — `time_mask`, `freq_mask` (SpecAugment), enum-dispatched `SpecAugOp` + `SpecAugPipeline::push` builder
  - [x] `time_warp.rs` — `time_warp`: single-anchor bilinear time-axis warping (no-op when T ≤ 2·max_w)

- [x] **Speaker** (speaker/) — 3 files
  - [x] `stats_pool.rs` — `stats_pool`: two-pass Bessel-corrected temporal mean+std pooling `[T,C] → [2C]`
  - [x] `attentive_pool.rs` — `AttentivePool`: bottleneck `tanh`-attention softmax over time → weighted mean+std `[2C]`
  - [x] `x_vector.rs` — `XVectorTdnn`: 5-layer dilated TDNN (Snyder 2018), stats pool, 512-d affine; `default_config()` + `tiny()`

- [x] **Integration tests** (lib.rs) — 21 E2E tests covering all modules + PTX × 6 SM versions

---

## Quality Gates

| Metric | Target | Achieved |
|--------|--------|----------|
| Compiler warnings | 0 | 0 |
| Clippy warnings | 0 | 0 |
| unwrap() in library code | 0 | 0 |
| C/Fortran build deps | 0 | 0 |
| Test count | >500 | 9,776+ |
| Test pass rate | 100% | 100% |
| Code lines (SLoC) | >30K | ~330,000 |
| Crate count | 12 | 38 |
| GPU arch coverage | SM 7.5--10.0 | SM 7.5--10.0 |
| Pure Rust | 100% default features | 100% |

---

## Vol.22: Time-Series Forecasting Architectures [COMPLETE]

### oxicuda-timeseries (30 files, ~8,500 SLoC, 177 tests)

Pure-Rust time-series forecasting and classification library: TCN, NHiTS, PatchTST,
TimesNet, iTransformer, RevIN, series decomposition — zero CUDA SDK dependency.
Time-major `[T, C]` layout throughout; all variates channels-last.

- [x] **Error types** (error.rs) — `TsError` (18 variants): DimensionMismatch, ShapeMismatch, EmptyInput, InvalidSequenceLength, InvalidNumVariates, InvalidPatchLen, InvalidStride, InvalidKernelSize, InvalidDilation, InvalidNumHeads, HeadDimMismatch, InvalidEmbedDim, InvalidHorizon, InvalidPoolSize, InvalidTopK, WeightShapeMismatch, NonFinite, Internal

- [x] **Handle** (handle.rs) — `SmVersion`, `LcgRng` (Box-Muller normals, Fisher-Yates shuffle), `TsHandle::default_handle()` (SM 8.0, device 0, seed 42)

- [x] **PTX kernels** (ptx_kernels.rs) — 7 GPU kernels × 6 SM versions (75/80/86/90/100/120)
  - [x] `moving_average_ptx` — Strided centred moving average over time axis
  - [x] `patch_embed_1d_ptx` — Extract overlapping 1-D patches [N,T]→[N,num_patches,patch_len]
  - [x] `causal_temporal_conv_ptx` — Dilated causal 1-D conv for TCN residual blocks
  - [x] `auto_correlation_ptx` — FFT magnitude-squared step for Autoformer/TimesNet
  - [x] `revin_normalize_ptx` — RevIN normalise with per-(n,c) stats + learnable affine
  - [x] `multirate_pool_ptx` — Average pool at variable stride for NHiTS multi-rate sampling
  - [x] `period_detect_ptx` — Top-k FFT magnitude reduction for TimesNet period detection

- [x] **Normalisation** (norm/) — `RevIn` (reversible instance norm with forward+inverse, Bessel-corrected stats), `InstanceNorm1d` (per-variate instance norm with optional affine)

- [x] **Decomposition** (decomp/) — `MovingAvg` (centred, replicate-pad), `SeriesDecomp` (trend + seasonal split matching Autoformer)

- [x] **Patch embedding** (patch/) — `PatchEmbed1d` (overlapping 1-D patches, Xavier init, univariate + multivariate `forward_mv`)

- [x] **TCN** (tcn/) — `TcnBlock` (weight-normalised dilated causal conv, Kaiming He init, optional 1×1 residual projection), `TcnEncoder` (exponential dilation schedule 2^i, tiny/default configs)

- [x] **NHiTS** (nhits/) — `MultiRateSampler` (avg pool + nearest-neighbour upsample), `NHitsBlock` (pool→MLP→backcast+forecast heads), `NHits` (hierarchical residual stacks with pool_sizes=[1,2,4])

- [x] **PatchTST** (patchtst/) — `PatchTst` (channel-independent patches → sinusoidal PE → N×pre-LN TransformerLayer → per-variate linear head), `PatchTstConfig::tiny/base`

- [x] **TimesNet** (timesnet/) — `TimesBlock` (O(T²) DFT period detection → top-k 2-D reshape → depthwise 3×3 conv → weighted sum → residual + LN), `TimesNet` (input proj → blocks → flatten → linear head)

- [x] **iTransformer** (itransformer/) — `InvertedBlock` (attention over C variate tokens), `ITransformer` (variate embedding → N blocks → per-variate head), `ITransformerConfig::tiny/base`

- [x] **Forecasting heads** (head/) — `LinearHead` (in→out, batch + per-variate ts variants), `MlpHead` (in→hidden→out with ReLU, Kaiming init for layer 1)

- [x] **Integration tests** (lib.rs) — 20 E2E tests covering all modules + PTX × 6 SM versions
- [x] **Benchmarks** (benches/ts_ops.rs) — 7 PTX bench groups × 4 SM versions + 5 architecture forward benches

---

## Vol.23: Bayesian Deep Learning [COMPLETE]

### oxicuda-bayes (24 files, ~6,020 SLoC, 188 tests)

Pure-Rust Bayesian deep learning library: variational inference, Bayesian
layers, MC Dropout, Deep Ensembles, SWAG, last-layer Laplace, calibration
metrics and post-hoc recalibration — zero CUDA SDK dependency.

- [x] **Error types** (error.rs) — `BayesError` (16 variants): DimensionMismatch, EmptyInputs, InvalidDropoutRate, InvalidTemperature, InvalidPriorVariance, NonPositiveSigma, InsufficientSamples, InsufficientEnsembleMembers, CalibrationSetEmpty, NCalibBinsTooSmall, IsotonicNotMonotone, PlattFitFailed, TemperatureNotFinite, FlowDimensionMismatch, NanEncountered, Internal

- [x] **Handle** (handle.rs) — `SmVersion` with `ptx_version_str()` mapping sm≥100→"8.7" / sm≥90→"8.4" / sm≥80→"8.0" / else "7.5"; `LcgRng` with Box-Muller `next_normal_pair`/`fill_normal`/`shuffle`; `BayesHandle::default_handle()` (SM 8.0, device 0, seed 42)

- [x] **PTX kernels** (ptx_kernels.rs) — 7 GPU kernels × 6 SM versions (75/80/86/90/100/120)
  - [x] `kl_gaussian_ptx` — Per-element KL(N(μ,σ²) ‖ N(0,1)) with `ex2.approx`/`lg2.approx` and `atom.global.add.f32` accumulation
  - [x] `mc_dropout_mask_ptx` — Bernoulli dropout mask via inline LCG `(rand > drop) ? 1/keep : 0`
  - [x] `local_reparam_ptx` — Local reparameterisation with Box-Muller sampling
  - [x] `ece_bucket_ptx` — ECE histogram binning with atomic counters
  - [x] `ensemble_aggregate_ptx` — Ensemble mean/variance over M member logits
  - [x] `flipout_perturb_ptx` — Flipout ±1 sign perturbation for variance reduction
  - [x] `temp_scale_logits_ptx` — Temperature scaling of logits

- [x] **Bayesian layers** (layers/)
  - [x] `BayesLinear` (bayes_linear.rs) — Bayes-by-Backprop linear with `softplus(rho)` σ parameterisation; `forward_sample` + `forward_kl`; per-weight prior N(0, σ²_prior)
  - [x] `BayesConv2d` (bayes_conv.rs) — same BBB scheme for spatial conv2d kernels
  - [x] `FlipoutLinear` / `FlipoutConv2d` (flipout.rs) — Flipout (Wen 2018) ±1 sign perturbation for in-batch decorrelation

- [x] **Variational inference** (variational/)
  - [x] `kl_gaussian` / `kl_gaussian_vec` (elbo.rs) — closed-form KL(q‖N(0,1)) and ELBO/IWAE objectives
  - [x] `MeanFieldDist` (mean_field.rs) — factored Gaussian; entropy, KL, ELBO, sample, sample_n
  - [x] `gaussian_sample` / `laplacian_sample` / log-prob (reparam.rs) — reparameterisation with straight-through estimator
  - [x] `PlanarFlow`, `RadialFlow` (flows.rs) — invertible 1-step normalising flows with log-det Jacobian

- [x] **Calibration** (calibration/) — 4 files
  - [x] `metrics.rs` — `expected_calibration_error` (ECE), `maximum_calibration_error` (MCE), `adaptive_calibration_error` (ACE) with quantile bins, `brier_score`, `negative_log_likelihood`, `top1_confidences`, `ReliabilityDiagram`/`ReliabilityBin`
  - [x] `temperature.rs` — `TemperatureScaler` with golden-section search NLL minimisation; argmax-preserving recalibration (Guo 2017)
  - [x] `isotonic.rs` — `IsotonicRegressor` Pool Adjacent Violators with weighted variant for non-parametric monotone recalibration
  - [x] `platt.rs` — `PlattScaler` two-parameter logistic recalibration with Lin et al. 2007 stable-target Newton + line search

- [x] **Uncertainty quantification** (uncertainty/) — 5 files
  - [x] `mc_dropout.rs` — `mc_dropout_predict` and `McDropoutPredictor` with Welford online mean/variance over T forward passes (Gal & Ghahramani 2016)
  - [x] `deep_ensemble.rs` — `DeepEnsemble` with `aggregate()` and `aggregate_probabilities()` (mean + sample variance with Bessel correction); `EnsembleStats`
  - [x] `swag.rs` — `SwagPosterior` with running first/second moments + FIFO low-rank deviation buffer; `θ̃ = μ + (1/√2)·σ_diag⊙z₁ + (1/√(2(K-1)))·D·z₂` sampling (Maddox 2019)
  - [x] `laplace.rs` — `LastLayerLaplace` with diagonal Hessian fit for binary logistic; closed-form predictive logit and probit-approximated marginal probability (MacKay 1992; Daxberger 2021)
  - [x] `entropy.rs` — `predictive_entropy`, `aleatoric_entropy`, `mutual_information` (BALD), `epistemic_entropy` (Houlsby 2011 decomposition)

- [x] **Integration tests** (lib.rs) — 12 E2E tests covering temperature scaling, isotonic, Platt, MC Dropout, Deep Ensemble, SWAG sampling, Laplace marginal, BALD, Brier+NLL, reliability diagram, and PTX kernels × 6 SM versions
- [x] **Benchmarks** (benches/bayes_ops.rs) — 7 PTX kernel groups × 4 SM versions + temperature_scaling_fit + isotonic_pav_fit + ece_compute + swag_sample + deep_ensemble_aggregate

---

## Vol.24: Federated Learning [COMPLETE]

### oxicuda-federated (26 files, ~4,630 SLoC, 145 tests)

Pure-Rust federated learning library: server algorithms (FedAvg/FedProx/SCAFFOLD/FedAdam),
gradient compression, differential privacy mechanisms with RDP/Moments accountants,
secure aggregation (Shamir + pairwise masking), and client selection — zero CUDA SDK dependency.

- [x] **Error types** (error.rs) — `FedError` (~17 variants): NoClients, EmptyGradient, DimensionMismatch, InvalidEpsilon, InvalidNoiseScale, InvalidThreshold, ShamirReconstructFailed, InsufficientShares, InvalidLearningRate, InvalidComprRank, NumberOfClientsBelowMinimum, NanEncountered, Internal, …

- [x] **Handle** (handle.rs) — `SmVersion`, `LcgRng` (Box-Muller, Fisher-Yates, Gaussian/Laplace samplers), `FedHandle::default_handle()`

- [x] **PTX kernels** (ptx_kernels.rs) — 7 GPU kernels × 6 SM versions
  - [x] `aggregate_mean_ptx` — Average across `K` client gradient buffers
  - [x] `dp_clip_gradient_ptx` — Per-sample L2-norm gradient clipping for DP-SGD
  - [x] `fedavg_weighted_sum_ptx` — Sample-count-weighted FedAvg server update
  - [x] `gaussian_noise_ptx` — Box-Muller Gaussian noise for DP mechanism
  - [x] `pairwise_mask_ptx` — Pairwise additive mask for secure aggregation
  - [x] `qsgd_quantize_ptx` — QSGD stochastic quantisation with dithering
  - [x] `topk_mask_ptx` — Top-K sparsification mask via threshold

- [x] **Server algorithms** (algorithm/)
  - [x] `FedAvgConfig` / `FedAvgState` (fedavg.rs) — Sample-weighted parameter averaging (McMahan 2017)
  - [x] `FedProxConfig` (fedprox.rs) — Proximal regularisation `μ/2·‖θ−θ_global‖²` for client drift control (Li 2020)
  - [x] `ScaffoldClientState` / `ScaffoldState` (scaffold.rs) — Control variates `c_i, c` correcting client drift (Karimireddy 2020)
  - [x] `FedAdamState` (fedadam.rs) — Server-side Adam with momentum and AMSGrad option (Reddi 2021)

- [x] **Compression** (compression/)
  - [x] `PowerSgdCompressor` (powersgd.rs) — Low-rank power-iteration compression with error feedback (Vogels 2019)
  - [x] `stochastic_quantize` (quantize.rs) — QSGD bit-budget quantisation with dithering
  - [x] `random_sparsify` (randomk.rs) — RandomK sparsification with deterministic compression ratio
  - [x] `topk_sparsify` (topk.rs) — TopK magnitude sparsification with `error_feedback`

- [x] **Differential privacy** (privacy/)
  - [x] `GaussianMechanism` (gaussian.rs) — Calibrated Gaussian noise for L2-bounded queries
  - [x] `LaplacianMechanism` (laplacian.rs) — Calibrated Laplace noise for L1-bounded queries
  - [x] `MomentsAccountant` (moments.rs) — Moments accountant for DP-SGD ε-tracking (Abadi 2016)
  - [x] `rdp_gaussian` / `rdp_to_dp` / `compose_rdp` (rdp.rs) — Rényi differential privacy with conversion to (ε, δ)-DP
  - [x] `PateConfig` / `noisy_voting` / `data_dependent_epsilon` (pate.rs) — PATE student-teacher voting (Papernot 2017)

- [x] **Secure aggregation** (secure_agg/)
  - [x] `ShamirConfig` / `share_scalar` / `share_gradient` / `reconstruct_*` (shamir.rs) — Shamir (k, n) secret sharing over a Mersenne-prime field
  - [x] `generate_mask` / `apply_pairwise_masks` / `unmask` (masking.rs) — Bonawitz-style additive masking that cancels in aggregation
  - [x] `SecureAggregator` (aggregator.rs) — Drives the masked-then-aggregate flow

- [x] **Client selection** (selection/)
  - [x] `random_select` / `stratified_select` (random.rs) — Uniform random and stratified selection across client cohorts

- [x] **Integration tests** (lib.rs) — 10 E2E tests: FedAvg mean recovery, FedProx proximal term, Top-K + error feedback compensation, QSGD unbiased estimator, Gaussian DP noise calibration, RDP linear composition, Shamir scalar/gradient round-trip, random_select uniqueness, PTX × 6 SM versions

- [x] **Benchmarks** (benches/fed_ops.rs) — 7 PTX kernel groups × 4 SM versions + fedavg_aggregate + topk_sparsify + qsgd_quantize + shamir share/reconstruct

---

## Vol.25: Neural Architecture Search [COMPLETE]

### oxicuda-nas (21 files, ~3,736 SLoC, 63 tests)

Pure-Rust NAS library: differentiable architecture search (DARTS), evolutionary
multi-objective search (NSGA-II), one-shot supernets with weight-sharing, and
slimmable networks — zero CUDA SDK dependency.

- [x] **Error types** (error.rs) — `NasError` (~14 variants): EmptyPopulation, InvalidArchEncoding, OpKindOutOfRange, InvalidArchitectureWeights, InvalidGumbelTemperature, InvalidWidthMultiplier, MixedOpDimensionMismatch, MissingPrimitive, NumObjectivesMismatch, InvalidPopulationSize, NanEncountered, Internal, …

- [x] **Handle** (handle.rs) — `SmVersion`, `LcgRng`, `NasHandle::default_handle()`

- [x] **PTX kernels** (ptx_kernels.rs) — 7 GPU kernels × 6 SM versions
  - [x] `arch_grad_ptx` — Architecture parameter gradient accumulation
  - [x] `arch_softmax_ptx` — Stable softmax over `K` operation-mixing weights
  - [x] `crossover_uniform_ptx` — Uniform crossover for evolutionary mutations
  - [x] `flops_accumulate_ptx` — FLOP-cost accumulation across operations
  - [x] `gumbel_softmax_ptx` — Gumbel-softmax differentiable categorical sampling
  - [x] `mixed_op_blend_ptx` — Convex combination of operation outputs
  - [x] `pareto_dominate_ptx` — Pareto dominance check for multi-objective sort

- [x] **Operations** (ops/)
  - [x] `OpKind` / `OpWeights` (primitives.rs) — 8 standard DARTS primitives: skip, sep_conv 3×3/5×5, dil_conv 3×3/5×5, max/avg pool 3×3, none
  - [x] `MixedOp` (mixed_op.rs) — `out = Σ_k softmax(α)_k · op_k(x)` differentiable mixture
  - [x] `SearchSpace`, `CellSpace`, `NetworkSpace` (search_space.rs) — DARTS-style cell + network spaces

- [x] **DARTS** (darts/)
  - [x] `DartsCell` (cell.rs) — Multi-step cell with `K` candidate ops on each edge
  - [x] `DartsNetwork` (network.rs) — Stacked cells (normal + reduction) with auxiliary head
  - [x] `BilevelOptimizer` (bilevel.rs) — Bi-level w/α optimisation: weights on inner train loss, architecture on outer val loss
  - [x] `DiscretizedCell` / `DiscretizedNetwork` / `derive_discrete_cell` / `derive_network` (derive.rs) — Top-2 op selection and architecture derivation

- [x] **Evolutionary** (evolution/)
  - [x] `ArchEncoding` (encoding.rs) — Discrete genome representation
  - [x] `Population` (population.rs) — Population container with crossover/mutation operators
  - [x] `Individual` / `fast_non_dominated_sort` / `crowding_distance` / `nsga2_select` / `tournament_select` (nsga2.rs) — NSGA-II multi-objective EA (Deb 2002)

- [x] **Supernet** (supernet/)
  - [x] `Supernet` (weight_share.rs) — Weight-shared one-shot supernet (Bender 2018)
  - [x] `PathSampler` / `SamplingStrategy` (path_sample.rs) — Uniform / fairness-aware path sampling for SPOS / FairNAS
  - [x] `SlimmableNet` / `BnStats` / `WIDTH_MULTIPLIERS` (slimmable.rs) — Slimmable networks with per-width batch norm statistics (Yu 2019)

- [x] **Predictor** (predictor/)
  - [x] `LayerSpec` / `ArchFeatures` (predictor_io.rs) — Shared `[op-one-hot ‖ in_ch ‖ out_ch ‖ h ‖ w]` feature extractor used by all predictors
  - [x] `OpCost` / `op_cost` / `total_cost` (flops.rs) — Analytic FLOP + parameter accountant (sep/dilated conv `2·K²·C_in·HW + 2·C_in·C_out·HW`, pooling `9·C_out·HW`)
  - [x] `LatencyLut` (latency.rs) — Hardware-calibrated `(op, c_in, c_out, h, w)` lookup with default fallback
  - [x] `LatencyMlp` (latency.rs) — Two-layer ReLU MLP latency surrogate trained via per-sample MSE gradient descent
  - [x] `KnnAccuracyPredictor` (accuracy.rs) — Inverse-distance-weighted k-NN regression on architecture features
  - [x] `RbfAccuracyPredictor` (accuracy.rs) — Gaussian-kernel ridge regressor with closed-form Gauss-Jordan solve

- [x] **Integration tests** (lib.rs) — 5 E2E tests covering FLOP accountant, LUT predict, MLP train/predict, k-NN round-trip, RBF constant-target

- [x] **Benchmarks** (benches/nas_ops.rs) — 7 PTX kernel groups × 4 SM versions + population_random + nsga2_select + path_sample + mixed_op_blend

---

## Vol.26: Self-Supervised Learning [COMPLETE]

### oxicuda-ssl (25 files, ~4,277 SLoC, 150 unit + 12 E2E tests)

Pure-Rust self-supervised learning library covering the four canonical families:
contrastive (SimCLR, MoCo), non-contrastive (BYOL, Barlow Twins, VICReg),
masked (MAE), and clustering (SwAV, DINO). Plus shared infrastructure for
momentum encoders, projection / predictor heads, and SSL-style data
augmentation. Zero CUDA SDK dependency.

- [x] **Error types** (error.rs) — `SslError` (16 variants): DimensionMismatch, EmptyInput, InvalidTemperature, InvalidMomentum, InvalidMaskRatio, InvalidNumCrops, InvalidLossWeight, QueueCapacityTooSmall, QueueEmpty, NumPrototypesTooSmall, SinkhornDiverged, InvalidFeatureDim, BatchTooSmall, NanEncountered, InvalidProjectorDim, Internal

- [x] **Handle** (handle.rs) — `SmVersion`, `LcgRng` (Box-Muller normals, Fisher-Yates shuffle), `SslHandle::default_handle()` (SM 8.0, device 0, seed 42)

- [x] **PTX kernels** (ptx_kernels.rs) — 7 GPU kernels × 6 SM versions (75/80/86/90/100/120)
  - [x] `nt_xent_softmax_ptx` — Per-row stable softmax over `2N×2N` similarity matrix with `selp.f32` self-mask `-INF` on diagonal
  - [x] `momentum_update_ptx` — `θ_target = m·θ_target + (1-m)·θ_online` with `fma.rn.f32` and grid-stride loop
  - [x] `byol_cosine_loss_ptx` — `2 - 2·cos(p, sg(z))` per-element accumulation via `atom.global.add.f32`
  - [x] `barlow_cross_corr_ptx` — Cross-correlation matrix `C[i,j] = Σ_n Z_A[n,i]·Z_B[n,j]` with 2-D grid + atomic accumulate
  - [x] `random_mask_ptx` — Bernoulli mask via inline LCG `(rand < drop_ratio) ? 0 : 1` for MAE patch dropping
  - [x] `cosine_similarity_ptx` — Per-pair cosine similarity for memory-bank lookup with `atom.global.add.f32`
  - [x] `gather_features_ptx` — Memory-queue gather `out[k,d] = queue[idx[k], d]` for MoCo

- [x] **Contrastive** (contrastive/)
  - [x] `info_nce_loss` (info_nce.rs) — Symmetric InfoNCE with stable log-sum-exp; returns `(loss, accuracy@1)`
  - [x] `simclr_loss` / `SimClrConfig` (simclr.rs) — Symmetric NT-Xent at temperature τ=0.1 (Chen 2020)
  - [x] `MocoQueue` / `moco_loss` (moco.rs) — FIFO circular queue + InfoNCE with positive vs queue negatives (He 2020)

- [x] **Non-contrastive** (non_contrastive/)
  - [x] `byol_loss` / `ByolPredictor` (byol.rs) — L2-normalised cosine `2 - 2·cos(p, sg(z))` (Grill 2020)
  - [x] `barlow_twins_loss` / `BarlowTwinsConfig` (barlow.rs) — Cross-correlation `Σ(1-C_ii)² + λ·Σ_{i≠j} C_ij²` after column standardisation (Zbontar 2021)
  - [x] `vicreg_loss` / `VicRegConfig` (vicreg.rs) — Variance hinge + invariance MSE + off-diagonal covariance penalty (Bardes 2022)

- [x] **Masked** (masked/)
  - [x] `random_patch_mask` / `mae_reconstruction_loss` / `MaeConfig` (mae.rs) — Fisher-Yates patch selection (exact ratio) + masked-patch-only MSE; default mask ratio 0.75 (He 2022)

- [x] **Clustering** (clustering/)
  - [x] `swav_loss` / `sinkhorn_knopp` / `SwavConfig` (swav.rs) — Sinkhorn-Knopp normalised codes + swapped CE (3 iters default, ε=0.05, τ=0.1) (Caron 2020)
  - [x] `dino_loss` / `update_dino_centre` / `DinoConfig` (dino.rs) — Centred + sharpened student-teacher CE (τ_s=0.1, τ_t=0.04) (Caron 2021)

- [x] **Augment** (augment/)
  - [x] `color_jitter` / `random_grayscale_chw` (color.rs) — Per-channel multiplicative jitter + BT.601 grayscale conversion on `[3, H, W]` images
  - [x] `multi_crop` / `MultiCropConfig` / `CropSpec` (multi_crop.rs) — DINO/SwAV global+local crop spec generation (default 2 globals @ 224 + 6 locals @ 96)

- [x] **Momentum** (momentum/)
  - [x] `EmaUpdater` / `cosine_momentum` (ema.rs) — Element-wise EMA target update + half-cosine momentum schedule (BYOL: 0.996 → 1.0)

- [x] **Head** (head/)
  - [x] `MlpProjector` (projector.rs) — 2-layer Linear→ReLU→Linear projection head with Kaiming init; supports per-sample and batched forward
  - [x] `PredictorHead` (predictor.rs) — Identical architecture used as the additional predictor on the online branch in BYOL/SimSiam

- [x] **Integration tests** (lib.rs) — 12 E2E tests: SimCLR aligned-pair drop, MoCo queue lifecycle, BYOL identity = 0, Barlow finite, VICReg combine, MAE mask ratio, Sinkhorn uniform, DINO centred CE, EMA monotone, MLP projector shape, multi_crop count, PTX kernels × 6 SM versions

- [x] **Benchmarks** (benches/ssl_ops.rs) — 7 PTX kernel groups × 4 SM versions + 5 algorithm benches: simclr_loss_b64_d128, moco_loss_b16_d64_q256, barlow_loss_b256_d64, mae_mask_p196_r075, dino_loss_b64_k128

---

## Vol.27: Adversarial Robustness [COMPLETE]

### oxicuda-adversarial (12 files, ~4,943 SLoC, 165 tests)

Pure-Rust adversarial robustness library covering both the attack side (FGSM,
PGD L∞/L2, MIM, CW, AutoPGD) and the defence side (TRADES, MART, Randomized
Smoothing, IBP/certified bounds). Includes Lp-ball threat-model primitives,
ε-budget tracking, and robustness evaluation metrics. Zero CUDA SDK dependency.

- [x] **Error types** (error.rs) — `AdvError` (15 variants): DimensionMismatch, EmptyInput, InvalidEpsilon, InvalidAlpha, InvalidNumSteps, InvalidLpNorm, InvalidTemperature, InvalidNoiseSigma, InvalidConfidence, InsufficientCertSamples, InvalidLossWeight, BudgetExceeded, NanEncountered, OptimizationDiverged, AttackFailedAll, Internal

- [x] **Handle** (handle.rs) — `SmVersion`, `LcgRng` (Box-Muller normals, Fisher-Yates shuffle, uniform `[0,1)`, Knuth MMIX 64-bit LCG), `AdvHandle::default_handle()` (SM 8.0, device 0, seed 42)

- [x] **PTX kernels** (ptx_kernels.rs) — 7 GPU kernels × 6 SM versions (75/80/86/90/100/120)
  - [x] `fgsm_step_ptx` — `x_adv[i] = clamp(x[i] + ε·sign(grad[i]), lo, hi)` with `fma.rn.f32` and grid-stride loop
  - [x] `pgd_proj_l_inf_ptx` — L∞ projection: `out[i] = clamp(clamp(x[i], x_orig[i]−ε, x_orig[i]+ε), lo, hi)`
  - [x] `pgd_proj_l2_ptx` — L2 projection: scale `δ = x − x_orig` so `‖δ‖₂ ≤ ε` via host-supplied norm + `div.rn.f32`
  - [x] `smoothing_noise_ptx` — Gaussian noise `z ~ N(0, σ²)` via inline LCG + Box-Muller (`lg2.approx.f32` / `cos.approx.f32`)
  - [x] `grad_sign_ptx` — `out[i] = sign(grad[i])` (`+1 / 0 / −1`) using `selp.f32` double-predicate
  - [x] `certified_radius_reduce_ptx` — Per-block argmax over `[K]` class-count vector for smoothed-predictor read-off
  - [x] `attack_loss_grad_ptx` — `out[i] = clamp(x[i] + α·dir[i], lo, hi)` inner step for MIM/PGD with momentum-accumulated gradient

- [x] **Attacks** (attacks/)
  - [x] `fgsm_attack` (fgsm.rs) — Single-step Fast Gradient Sign Method (Goodfellow 2014): `x_adv = clamp(x + ε·sign(∇L), lo, hi)`
  - [x] `pgd_attack_l_inf` / `pgd_attack_l2` / `PgdConfig` (pgd.rs) — Projected Gradient Descent with random restart and L∞/L2 projections (Madry 2018)
  - [x] `mim_attack` / `MimConfig` (mim.rs) — Momentum Iterative Method with exponential momentum accumulation (Dong 2018)
  - [x] `cw_attack` / `CwConfig` (cw.rs) — Carlini-Wagner L2 attack with binary-search confidence parameter and change-of-variable tanh reparametrisation (Carlini 2017)
  - [x] `auto_pgd_attack` / `AutoPgdConfig` (auto_pgd.rs) — AutoPGD with step-size schedule and checkpointing (Croce 2020)

- [x] **Defenses** (defenses/)
  - [x] `trades_loss` / `TradesConfig` (trades.rs) — TRADES regulariser: CE(clean) + β·KL(clean ‖ adv) with KL computed from log-softmax pairs (Zhang 2019)
  - [x] `mart_loss` / `MartConfig` (mart.rs) — MART: misclassification-aware adversarial training with BCE on natural examples + weighted KL term (Wang 2020)
  - [x] `smoothed_predict` / `certified_radius` / `RsConfig` (randomized_smoothing.rs) — Cohen (2019) randomized smoothing: Monte-Carlo majority vote + Binomial CI for certified L2 radius
  - [x] `ibp_propagate` / `lipschitz_certified_radius` / `IntervalBound` (certified_bounds.rs) — Interval Bound Propagation through affine layers with per-bound `relu()`; Lipschitz certified radius `m / (L·√2)`

- [x] **Threat model** (threat_model/)
  - [x] `LpNorm` / `l_inf_norm` / `l1_norm` / `l2_norm` / `project_l_inf` / `project_l2` (lp_ball.rs) — Lp-ball norm computations and projections (L1 / L2 / L∞)
  - [x] `EpsilonBudget` (budget.rs) — ε-budget tracker with `spend()` / `remaining()` and `BudgetExceeded` error on overdraft

- [x] **Metrics** (metrics/)
  - [x] `robust_accuracy` (robust_accuracy.rs) — Fraction of adversarial examples predicted correctly; complement = attack success rate
  - [x] `certified_accuracy` (robust_accuracy.rs) — Fraction of examples both correctly predicted and certified at radius ≥ threshold
  - [x] `attack_success_rate` (asr.rs) — Fraction of adversarial examples on which the attack successfully changed the prediction

- [x] **Integration tests** (lib.rs) — 12 E2E tests: FGSM pushes away from target, PGD L∞/L2 respect ε-ball, MIM with zero-decay matches PGD, TRADES collapses to CE when clean=adv, MART loss finite, RS constant classifier returns top class, IBP propagates through ReLU, Lipschitz radius formula, robust/certified accuracy, PTX kernels × 6 SM versions, ε-budget lifecycle

- [x] **Benchmarks** (benches/adv_ops.rs) — 7 PTX kernel groups × 4 SM versions + 4 algorithm benches: fgsm_attack_d1024, pgd_l_inf_attack_d512_n10, trades_loss_b64_k10, ibp_propagate_64x32

---

## Vol.28: Multi-Modal Learning [COMPLETE]

### oxicuda-multimodal (23 files, ~6,149 SLoC, 156 tests)

Pure-Rust multi-modal learning primitives covering cross-modal attention, compact bilinear fusion (MLB/MFB), contrastive alignment (CLIP bidirectional InfoNCE, ImageBind triple alignment), ITM head, BERT text encoder, ViT image encoder, Conformer audio encoder, temporal ViT video encoder, prefix-LM captioning, and VQA head. Zero CUDA SDK dependency.

- [x] **Error types** (error.rs) — `MultiModalError` (14 variants): DimensionMismatch, EmptyInput, InvalidNHeads, InvalidDModel, InvalidVocabSize, InvalidMaxSeqLen, InvalidImageSize, InvalidPatchSize, InvalidAudioFeatures, InvalidVideoFrames, NanEncountered, BatchSizeMismatch, InvalidAnswerCount, Internal

- [x] **Handle** (handle.rs) — `SmVersion`, `LcgRng` (Knuth MMIX 64-bit LCG), `MultiModalHandle::default_handle()` (SM 8.0, device 0, seed 42)

- [x] **PTX kernels** (ptx_kernels.rs) — 7 GPU kernels × 6 SM versions (75/80/86/90/100/120):
  - [x] `cross_attn_score_ptx` — QKᵀ/√d_k scaled dot-product with `fma.rn.f32`
  - [x] `modal_align_loss_ptx` — bidirectional InfoNCE over batch diagonal
  - [x] `bilinear_pool_ptx` — tanh Hadamard product for MLB/MFB compact bilinear pooling
  - [x] `temporal_pool_ptx` — mean-pool across frames, `atom.global.add.f32`
  - [x] `token_merge_ptx` — concatenate and project tokens for prefix-LM
  - [x] `gate_fusion_ptx` — softmax-gated attention fusion across modalities
  - [x] `itm_bce_ptx` — numerically stable BCE for image-text matching

- [x] **Cross-modal attention** (cross_attn/)
  - [x] `CrossAttention` (cross_attention.rs) — MHSA with Q from modality A, K/V from modality B; scaled dot-product attention with softmax and output projection
  - [x] `SelfCrossBlock` (self_cross_block.rs) — pre-norm residual block: LN→self-attn→LN→cross-attn→LN→FFN, each with skip connection

- [x] **Fusion** (fusion/)
  - [x] `ConcatFusion` (concat_fusion.rs) — concatenate embeddings then project to joint space
  - [x] `MlbFusion` / `MfbFusion` (bilinear_fusion.rs) — compact bilinear pooling: MLB = tanh(Wv·v) ⊙ tanh(Wq·q) → linear; MFB = expand-to-k×d, sum-pool pairs, tanh
  - [x] `AttentionFusion` (attention_fusion.rs) — softmax over modality-specific keys → weighted sum of value embeddings

- [x] **Alignment** (alignment/)
  - [x] `clip_loss` (contrastive.rs) — L2-normalized bidirectional InfoNCE loss (symmetric cross-entropy over similarity matrix)
  - [x] `imagebind_loss` (contrastive.rs) — triple alignment loss over three modalities via pairwise InfoNCE average
  - [x] `ItmHead` / `itm_loss` (matching.rs) — 2-layer MLP binary classifier for image-text matching; numerically stable BCE

- [x] **Encoders** (encoder/)
  - [x] `BertEncoder` (text_encoder.rs) — token+pos embedding → N×(self-attn+FFN+LN) transformer blocks → CLS-pool → d_model-dim output
  - [x] `ViTEncoder` (image_encoder.rs) — flatten patches → linear embed → CLS prepend → pos embed → N transformer blocks → CLS-pool
  - [x] `AudioEncoder` (audio_encoder.rs) — linear mel projection → N Conformer blocks (conv+attn+FFN) → statistics pooling (mean‖std) → 2×d_model
  - [x] `VideoEncoder` (video_encoder.rs) — spatial ViT per frame → temporal attention → mean-pool → d_model

- [x] **Captioning/VQA** (caption/)
  - [x] `PrefixLm` (prefix_lm.rs) — greedy autoregressive decoding with visual prefix cross-attention; configurable max length
  - [x] `VqaHead` / `vqa_loss` (vqa_head.rs) — 2-layer MLP over fused features → n_answers logits; cross-entropy loss

- [x] **Integration tests** (lib.rs) — 12 E2E tests: CLIP loss on L2-normalized features ≈ ln(N), ImageBind triple loss finite, cross-attention output shape, MlbFusion output shape, ItmHead BCE loss decreasing, BertEncoder CLS pool shape, ViTEncoder tiny forward, AudioEncoder stats-pool shape, VideoEncoder temporal mean-pool, PrefixLm greedy decode terminates, VqaHead logits shape, PTX kernels × 6 SM versions

- [x] **Benchmarks** (benches/mm_ops.rs) — 7 PTX kernel groups × 4 SM versions + 5 algorithm benches: clip_loss_b64_d256, mlb_fusion_b32_d512, cross_attn_heads8_d64_len32, bert_tiny_forward, vit_tiny_forward

---

## Vol.29: Continual Learning [COMPLETE]

### oxicuda-continual (25 files, ~5,037 SLoC, 165 tests)

Pure-Rust continual and lifelong learning library covering all major families of catastrophic-forgetting mitigation: regularization (EWC/SI/MAS), architecture (PackNet/Piggyback/Progressive NN), and experience replay (ER/GEM/A-GEM/DER++). Also includes forgetting/plasticity metrics and task-incremental / class-incremental data streams. Zero CUDA SDK dependency.

- [x] **Error types** (error.rs) — `ContinualError` (15 variants): DimensionMismatch, EmptyInput, InvalidLambda, InvalidBufferCapacity, InvalidTaskId, InsufficientData, InvalidThreshold, InvalidAlpha, InvalidBeta, GemProjectionFailed, NanEncountered, InvalidMaskSparsity, InvalidLateralDim, StreamExhausted, Internal

- [x] **Handle** (handle.rs) — `SmVersion`, `LcgRng` (Knuth MMIX 64-bit LCG, reservoir sampling, Box-Muller), `ContinualHandle::default_handle()` (SM 8.0, device 0, seed 42)

- [x] **PTX kernels** (ptx_kernels.rs) — 7 GPU kernels × 6 SM versions (75/80/86/90/100/120):
  - [x] `ewc_penalty_ptx` — `fma.rn.f32` for λ/2·Σ F_i·(θ_i−θ*_i)², `atom.global.add.f32`
  - [x] `fisher_diag_ptx` — element-wise `g²` accumulate for empirical Fisher diagonal
  - [x] `gradient_project_ptx` — half-space projection `g − (g·m / m·m)·m` for GEM
  - [x] `mask_apply_ptx` — `w *= mask` with `setp.ne.u32` predicate for PackNet/Piggyback
  - [x] `si_omega_update_ptx` — `|Δθ·∇L|` synaptic importance accumulate
  - [x] `logit_distill_ptx` — KL divergence via `ex2/lg2` approximations for DER++
  - [x] `replay_sample_ptx` — reservoir sampling conditional swap via LCG

- [x] **Regularization** (regularization/)
  - [x] `EwcRegularizer` (ewc.rs) — Elastic Weight Consolidation (Kirkpatrick 2017): empirical Fisher diagonal `F_i = (1/N)Σg_i²`; penalty `λ/2·Σ_t Σ_i F_i^t·(θ_i−θ_i^{*t})²`; `add_task()` anchors a new task
  - [x] `SiState` (si.rs) — Synaptic Intelligence (Zenke 2017): online importance `Ω_i += |Δθ_i·∇L_i|`; SI penalty normalized by `(ΔΘ_i²+ξ)`
  - [x] `MasImportance` (mas.rs) — Memory-Aware Synapses (Aljundi 2018): momentum-weighted importance update `Ω = α·Ω + (1−α)·|∇L|`

- [x] **Architecture** (architecture/)
  - [x] `PackNetMask` (packnet.rs) — L1 magnitude pruning to sparsity fraction; task-specific binary masks; freeze pruned weights
  - [x] `PiggybackMask` (piggyback.rs) — real-valued mask → binary via threshold; effective weights `w_eff = w_base ⊙ binarize(m)`
  - [x] `ProgNnNetwork` (progressive.rs) — Progressive Neural Networks (Rusu 2016): frozen previous columns with lateral connections `h_k^l = relu(W·h + Σ U·h_prev)`

- [x] **Replay** (replay/)
  - [x] `ErBuffer` (er.rs) — Experience Replay with reservoir sampling (Vitter 1985): uniform buffer replacement with probability `capacity/n_seen`; Fisher-Yates batch sampling
  - [x] `GemMemory` / `gem_project_gradient` (gem.rs) — Gradient Episodic Memory (Lopez-Paz 2017): iterative half-space projection onto `g·g_k ≥ −margin` constraints, most-violated-constraint first
  - [x] `a_gem_project` (a_gem.rs) — Averaged GEM (Chaudhry 2018): single projection onto average reference gradient `g_ref = (1/T)Σg_k`
  - [x] `DerBuffer` / `der_loss` (dark_exp.rs) — Dark Experience Replay++ (Buzzega 2020): α·MSE(z, z_stored) + β·CE(z, y); reservoir buffer with stored logits

- [x] **Metrics** (metrics/)
  - [x] `AccuracyMatrix` / `average_forgetting` / `backward_transfer` / `plasticity` (forgetting.rs) — standard CL metrics: BWT = `(1/(T−1))Σ_k(acc[T−1,k]−acc[k,k])`, forgetting = `max_j acc[j,k] − acc[T−1,k]`
  - [x] `forward_transfer` / `intransigence` (intransigence.rs) — FWT = `(1/(T−1))Σ_k(acc[k−1,k]−acc_random[k])`; intransigence = transfer gap to isolated task training

- [x] **Data streams** (stream/)
  - [x] `TaskStream` (task_stream.rs) — task-incremental stream: ordered task sequence with batch sampler
  - [x] `ClassIncStream` (class_stream.rs) — class-incremental stream with disjoint label spaces; `n_classes_seen()` grows monotonically

- [x] **Integration tests** (lib.rs) — 12 E2E tests: EWC penalty zero before anchoring, EWC penalty positive after anchoring, SI importance accumulates, MAS importance update converges, PackNet prune+mask round-trip, GEM projection on 2D example, DER++ loss finite, reservoir sampling fills buffer uniformly, AccuracyMatrix BWT/forgetting formulas, TaskStream next_task(), ClassIncStream advance, PTX kernels × 6 SM versions

- [x] **Benchmarks** (benches/continual_ops.rs) — 7 PTX kernel groups × 4 SM versions + 5 algorithm benches: ewc_loss_d1024, fisher_diag_accumulate, gem_project_d512, er_sample_b32, packnet_prune_d1024

---

## Vol.30: 3D Geometry & Point Clouds [COMPLETE]

### oxicuda-geometry3d (36 files, ~7,146 SLoC, 189 tests)

Pure-Rust 3D geometry and point-cloud deep-learning library covering sampling (FPS/random/voxel-downsample), neighborhood queries (k-NN/ball-query/KD-tree), point feature operations (gather/group/interp), architectures (PointNet/PointNet++/DGCNN/Point-Transformer), voxel ops (voxelization/sparse 3D conv), mesh distances (Chamfer/EMD-Sinkhorn/normal PCA), 3D Gaussian splatting primitives, and SE(3)/quaternion/ICP transforms. Zero CUDA SDK dependency.

- [x] **Error types** (error.rs) — `Geom3dError` (15 variants): DimensionMismatch, EmptyPointCloud, InvalidPointDim, InvalidK, InvalidRadius, InvalidVoxelSize, InvalidSampleCount, InvalidShCoefficients, InvalidQuaternion, IcpDidNotConverge, EmdDidNotConverge, InvalidTopology, NanEncountered, BatchSizeMismatch, Internal

- [x] **Handle** (handle.rs) — `SmVersion`, `LcgRng` (Knuth MMIX 64-bit LCG, `next_usize`, `next_f32`), `Geom3dHandle::default_handle()` (SM 8.0, device 0, seed 42)

- [x] **PTX kernels** (ptx_kernels.rs) — 7 GPU kernels × 6 SM versions (75/80/86/90/100/120):
  - [x] `farthest_point_sample_ptx` — per-point distance update + `atom.global.max.f32` argmax reduce
  - [x] `ball_query_ptx` — radius test `d²<r²` + bounded atomic counter per query
  - [x] `gather_points_ptx` — indexed feature gather with `mul.wide.u32` 64-bit offset
  - [x] `voxelize_ptx` — voxel index from `(p−o)/v`, `atom.global.add.f32` per channel + count
  - [x] `chamfer_distance_ptx` — tiled pairwise dist, warp-min reduce, `atom.global.min.f32`
  - [x] `gaussian_project_ptx` — 3D→2D Jacobian `J·Σ·Jᵀ` via `fma.rn.f32`
  - [x] `sh_eval_ptx` — spherical harmonic evaluation L=0..2 with precomputed constants as f32-hex literals

- [x] **Sampling** (sampling/)
  - [x] `farthest_point_sample` (farthest_point_sample.rs) — deterministic FPS with `idx[0]=0` init; `dist[i] = min(dist[i], d²_to_last)`; argmax as next seed
  - [x] `random_sample` (random_sample.rs) — partial Fisher-Yates without replacement via LcgRng
  - [x] `voxel_downsample` (voxel_downsample.rs) — HashMap voxel grid, emit centroids + first-original-index per bucket; sort for determinism

- [x] **Neighborhood** (neighborhood/)
  - [x] `knn` (knn.rs) — brute-force k-NN per query; returns `(indices, sq_dists)` row-major `[nq×k]`
  - [x] `ball_query` (ball_query.rs) — radius-limited search; `usize::MAX` sentinel for empty slots; returns `(indices, counts)` `[nq×k_max]`
  - [x] `KdTree` (kd_tree.rs) — recursive median-split build; `nearest`, `knn`, `radius_search`; best-first with AABB pruning

- [x] **Point feature ops** (pointops/)
  - [x] `gather_points` (gather_points.rs) — `[n×c]` + `[k]` indices → `[k×c]` with bounds check
  - [x] `group_features` (group_features.rs) — `[n×c]` + `[k×s]` indices → `[k×s×c]`
  - [x] `interp_features` (interp_features.rs) — 3-NN inverse-distance-weighted feature interpolation (ε=1e-10)

- [x] **Architectures** (arch/)
  - [x] `PointNet` (pointnet.rs) — T-Net (3×3 transform, identity-init) + shared MLP [3→64→128→1024] + global max-pool + FC head → class logits
  - [x] `SetAbstraction` / `FeaturePropagation` (pointnet_pp.rs) — PointNet++: FPS→ball-query→gather→MLP→max-pool; upsample via 3-NN interp + skip concat + MLP
  - [x] `EdgeConv` (dgcnn.rs) — DGCNN: dynamic kNN graph in feature space; edge feat = concat(x_i, x_j−x_i); MLP+max-pool
  - [x] `PointTransformerLayer` (point_transformer.rs) — vector self-attention with relative-position MLP encoding δ_ij; element-wise attention weights

- [x] **Voxel ops** (voxel/)
  - [x] `VoxelGrid` (voxelize.rs) — scatter points→grid with Mean/Max/Sum pooling; `occupied_centroids()` emission
  - [x] `SparseConv3d` (sparse_conv3d.rs) — Minkowski-style sparse 3D convolution; HashMap output accumulation; configurable kernel size

- [x] **Mesh distances** (mesh/)
  - [x] `chamfer_distance` / `chamfer_distance_grad` (chamfer_distance.rs) — bidirectional CD with gradient `2(a−b_nearest)/|A|`
  - [x] `earth_movers_distance` (earth_movers.rs) — entropy-regularized OT via log-domain Sinkhorn (clamp ±50, ε≥1e-3)
  - [x] `estimate_normals` (normal_estimate.rs) — per-point PCA normals via 3×3 covariance smallest-eigenvector; +z orientation

- [x] **Gaussian splatting** (gaussian/)
  - [x] `Gaussian3d` (gaussian.rs) — wxyz quaternion, log-scale, pre-sigmoid opacity, SH coefficients; `covariance3d()`, `sh_color()`
  - [x] `project_gaussian` (project.rs) — view-space projection, 2×2 covariance via Jacobian, low-pass `Σ_2d += 0.3·I`
  - [x] `rasterize_gaussians` (rasterize.rs) — depth-sort, 3σ AABB, alpha-composite front-to-back; T<1e-4 early termination

- [x] **Transforms** (transform/)
  - [x] `RigidTransform` (rigid.rs) — SE(3): rotation matrix + translation; Rodrigues axis-angle; compose, inverse, apply
  - [x] `Quat` (quaternion.rs) — wxyz quaternion; mul, conjugate, to/from rotation matrix; slerp with shortest-path sign-flip and lerp fallback
  - [x] `icp` (icp.rs) — point-to-point ICP via 3×3 Jacobi SVD, sign-correct `det(VUᵀ)`, KD-tree correspondences

- [x] **Integration tests** (lib.rs) — 12 E2E tests: FPS selects m distinct points, PointNet forward valid logits, SetAbstraction reduces point count, DGCNN output shape, chamfer(A,A)=0, ICP identity convergence, voxelize round-trip, Gaussian project valid depth, KD-tree nearest correctness, kNN vs brute-force agreement, LcgRng determinism, PTX kernels × 6 SM versions

- [x] **Benchmarks** (benches/geom3d_ops.rs) — 7 PTX kernel groups × 4 SM versions + 5 algorithm benches: fps_n4096_m512, knn_n2048_k16, chamfer_na1024_nb1024, pointnet_forward_n512, kdtree_build_n4096

---

## Vol.31: Physics-Informed Neural Networks [COMPLETE]

### oxicuda-pinn (36 files, ~6,730 SLoC, 264 tests)

Pure-Rust physics-informed scientific ML library covering: forward-mode autodiff (dual numbers, MultiDual), tape-based reverse-mode AD (Wengert list), PINN losses (residual/boundary/IC + NTK adaptive weighting), Neural ODEs (Euler/Heun/RK4/Dopri45 + continuous adjoint method + CNF + latent-ODE), neural operators (FNO 1D/2D, DeepONet, MWT, GNO), PDE templates (heat/wave/Burgers/Poisson/Navier-Stokes), coordinate-based MLP/SIREN networks, and adaptive collocation sampling (residual-adaptive/LHS/Halton). Zero CUDA SDK dependency.

- [x] **Error types** (error.rs) — `PinnError` (16 variants): DimensionMismatch, EmptyInput, InvalidStepSize, InvalidTimeInterval, NanEncountered, InvalidGridResolution, TooManyFourierModes, InvalidLayerWidth, InvalidNetworkDepth, InvalidWeight, InvalidActivation, SolverDivergence, EmptyCollocationSet, TapeIndexOutOfRange, InvalidPdeCoefficient, Internal

- [x] **Handle** (handle.rs) — `SmVersion`, `LcgRng` (Knuth MMIX 64-bit LCG, Box-Muller normals, `next_f32`, `next_usize`), `PinnHandle::default_handle()` (SM 8.0, device 0, seed 42)

- [x] **PTX kernels** (ptx_kernels.rs) — 7 GPU kernels × 6 SM versions (75/80/86/90/100/120):
  - [x] `pinn_residual_ptx` — `r² = r·r`, `atom.global.add.f32` reduction for `Σ|F|²`
  - [x] `spectral_conv_ptx` — complex multiply for FNO spectral convolution via `fma.rn.f32`
  - [x] `dual_op_ptx` — dual-number multiply `(a+εa')·(b+εb') = ab + ε(a'b+ab')` via 4× `fma.rn.f32`
  - [x] `adjoint_ode_ptx` — reverse-time Euler step `a[i] += h·dadt[i]` for adjoint accumulation
  - [x] `branch_trunk_dot_ptx` — DeepONet inner product with warp-shuffle `shfl.sync.bfly.b32` reduce
  - [x] `siren_forward_ptx` — `sin(ω₀·(Wx+b))` SIREN layer via `sin.approx.f32`
  - [x] `lhs_sample_ptx` — LCG per thread, cell-offset sample for Latin Hypercube Sampling

- [x] **Autodiff** (autodiff/)
  - [x] `Dual` (dual.rs) — forward-mode AD: dual numbers with all standard ops (sin/cos/exp/ln/sqrt/tanh/powi/abs + arithmetic ops) and chain rule
  - [x] `Tape` / `Var` (tape.rs) — reverse-mode AD via index-based Wengert list; `gradient()` reverse pass; ops: add/sub/mul/div/sin/cos/exp/tanh/sq
  - [x] `MultiDual<N>` (multidim.rs) — simultaneous N-variable partial derivatives; arithmetic and transcendental ops with product/chain rule on grad arrays

- [x] **PINN losses** (pinn_loss/)
  - [x] `pde_residual_loss` / `compute_residuals` (residual.rs) — MSE over `|F[u;θ](x_i)|²`; closure-based residual function
  - [x] `bc_loss` / `BcType` (boundary.rs) — Dirichlet / NeumannX / NeumannY boundary condition loss
  - [x] `ic_loss` (initial.rs) — initial condition MSE loss
  - [x] `AdaptiveWeights` (weighting.rs) — NTK-style λ update: `λ_i ← α·λ_i + (1−α)/||∇L_i||`; `weighted_loss()` combiner
  - [x] `CausalPinnLoss` / `CausalPinnConfig` (causal.rs) — Causal PINN training (Wang et al. 2022): exponentially-decaying temporal weights `w_i = exp(-ε·Σ_{j<i}r_j²)`; convergence criterion, partial loss, effective coverage, cumulative squared residuals
  - [x] `SaPinn` / `SaPinnConfig` (sa_pinn.rs) — Self-Adaptive PINN (McClenny & Braga-Neto 2021): per-point trainable weights via softplus activation; gradient-ascent `λ_i += lr·r_i²`; normalised/unnormalised modes; effective_n perplexity; quotient-rule lambda_gradient

- [x] **Neural ODE / SDE** (neural_ode/)
  - [x] `euler_step` / `heun_step` / `rk4_step` / `dopri45_step` / `integrate_fixed` / `integrate_adaptive` (solvers.rs) — Dormand-Prince RK4(5) with exact Butcher tableau coefficients; adaptive step control with `0.9·err^{-0.2}` rescaling
  - [x] `node_forward` / `node_adjoint_grad` (adjoint.rs) — continuous adjoint method: forward trajectory storage, reverse-time integration of `ȧ = −aᵀ·∂f/∂y`, accumulation of `dL/dθ = −∫aᵀ·∂f/∂θ dt`
  - [x] `cnf_forward` / `hutchinson_trace` / `dense_trace` (cnf.rs) — Continuous Normalizing Flow: log-density via Hutchinson Rademacher trace estimator or dense trace via dual numbers
  - [x] `LatentOde` (latent_ode.rs) — encoder GRU → reparametrize → ODE on latent → decoder MLP; Box-Muller normals via LcgRng

- [x] **Neural Operators** (neural_op/)
  - [x] `Fno1d` / `Fno2d` (fno.rs) — Fourier Neural Operator: inline O(N²) DFT (separable for 2D), spectral conv (complex multiply up to k_max modes), GeLU activation, lift/project layers
  - [x] `DeepONet` (deeponet.rs) — branch network (encodes function samples) × trunk network (encodes query coords) → inner product output; batch forward
  - [x] `Mwt` (mwt.rs) — Multiwavelet Transform Operator via Haar wavelet decompose/reconstruct per-level with learnable kernel
  - [x] `Gno` (gno.rs) — Graph Neural Operator: radius-based neighbor aggregation with kernel MLP `K(x_i−x_j; θ)·feat_j` → mean-pool

- [x] **PDE templates** (pde/)
  - [x] `heat_residual` / `heat_analytic` (heat.rs) — 1D heat: `∂u/∂t − α∂²u/∂x²`; analytic `sin(πx)·exp(−α·π²·t)`
  - [x] `wave_residual` / `wave_analytic` (wave.rs) — 1D wave: `∂²u/∂t² − c²∂²u/∂x²`; D'Alembert solution
  - [x] `burgers_residual` / `burgers_analytic` (burgers.rs) — 1D Burgers `∂u/∂t + u·∂u/∂x − ν∂²u/∂x²`; viscous shock tanh solution
  - [x] `poisson_residual` / `poisson_analytic` (poisson.rs) — 2D Poisson `∇²u=f`; `f=−2π²sin(πx)sin(πy) → u=sin(πx)sin(πy)`
  - [x] `ns_vorticity_residual` / `taylor_green_vortex` (navier_stokes.rs) — 2D NS vorticity form; Taylor-Green vortex `ω=2cos(x)cos(y)exp(−2νt)`

- [x] **Networks** (network/)
  - [x] `Mlp` / `Activation` (mlp.rs) — configurable MLP (tanh/sin/relu/gelu); SIREN init (first layer `U(−1/d, 1/d)`, hidden `U(−√(6/d)/ω₀, √(6/d)/ω₀)`); `grad_input()` via Tape AD; gradient descent `step()`
  - [x] `FourierFeatureNetwork` (coordinate_mlp.rs) — sinusoidal positional encoding `[sin(2πBx); cos(2πBx)]` with Gaussian random B (Box-Muller), then MLP

- [x] **Adaptive sampling** (sampling/)
  - [x] `residual_adaptive_sample` (residual_adaptive.rs) — importance sampling `p_i ∝ |R_i|^power` via inverse-CDF with LcgRng
  - [x] `latin_hypercube_sample` (latin_hypercube.rs) — LHS: each marginal cell hit exactly once via Fisher-Yates permutation per dimension
  - [x] `halton` / `halton_sequence` (quasi_random.rs) — Halton radical-inverse sequence using first d primes for low-discrepancy sampling

- [x] **Integration tests** (lib.rs) — 12 E2E tests: heat PINN loss computable, Burgers residual near-zero on analytic solution, NeuralODE RK4 exp-decay within 1e-4, adjoint gradient sign correct, FNO1D forward shape, FNO2D forward shape, DeepONet scalar output, CNF log-det finite, tape gradient x²=2x, dual sin(x²) dvalue, LHS marginal coverage, PTX kernels × 6 SM versions

- [x] **Benchmarks** (benches/pinn_ops.rs) — 7 PTX kernel groups × 4 SM versions + 5 algorithm benches: rk4_step_d64, dopri45_step_d32, fno1d_forward_n32, dft_n32, lhs_sample_d4_n256

---

## Vol.32: RLHF & Alignment Algorithms [COMPLETE]

### oxicuda-rlhf (21 files, ~5,500 SLoC, 25 tests)

Pure-Rust RLHF and alignment primitives covering all modern alignment algorithms: Direct Preference Optimization (DPO), Identity Preference Optimization (IPO), Kahneman-Tversky Optimization (KTO), Odds Ratio Preference Optimization (ORPO), Simple Preference Optimization (SimPO), Bradley-Terry reward modeling, reward normalization (Welford), PPO-RLHF rollout with GAE + KL penalty, adaptive KL controller, and masked SFT cross-entropy loss. Zero CUDA SDK dependency.

- [x] **Error types** (error.rs) — `RlhfError` (15 variants): DimensionMismatch, EmptyInput, InvalidBeta, InvalidTemp, NanEncountered, InvalidLambda, LogProbsRequired, MismatchedPairLength, InvalidMargin, KlDivergence, InvalidReferenceLogProb, RewardNormFailed, InvalidMaskValue, Internal, InvalidClipRatio

- [x] **Handle** (handle.rs) — `SmVersion`, `LcgRng` (Knuth MMIX 64-bit LCG), `RlhfHandle::default_handle()` (SM 8.0, device 0, seed 42)

- [x] **PTX kernels** (ptx_kernels.rs) — 7 GPU kernels × 6 SM versions (75/80/86/90/100/120):
  - [x] `bt_reward_loss_ptx` — `-Σ log(σ(r_chosen - r_rejected))` per pair
  - [x] `dpo_loss_ptx` — DPO: `-log σ(β·((lp_w - ref_w) - (lp_l - ref_l)))` per pair
  - [x] `ipo_loss_ptx` — IPO squared: `((lp_w - ref_w) - (lp_l - ref_l) - 1/(2β))²`
  - [x] `kto_loss_ptx` — KTO: desirable `(1-σ(β·(r-z₀)))` + undesirable with λ weights; z₀=ln2
  - [x] `orpo_odds_ptx` — ORPO log-odds: `log(exp(lp)/(1-exp(lp)+ε))` per sequence
  - [x] `rlhf_kl_ptx` — forward KL penalty per token: `exp(lp)·(lp - ref_lp)`
  - [x] `sft_mask_ptx` — masked cross-entropy per token; division by mask-sum in host code

- [x] **Preference data** (preference/)
  - [x] `PreferencePair` / `PairBatch` (pair.rs) — paired chosen/rejected log-probs + reference model log-probs; length validation
  - [x] `bt_reward_loss` / `RewardHead` (bradley_terry.rs) — Bradley-Terry pairwise loss `-E[log σ(r_w - r_l)]`; linear reward head with Xavier init

- [x] **Reward modeling** (reward/)
  - [x] `RewardModel` (model.rs) — multi-layer MLP with ReLU activations → scalar reward
  - [x] `RewardNormalizer` (normalize.rs) — Welford online mean/variance; `normalize()` whitens to zero-mean unit-variance

- [x] **Preference alignment losses** (dpo/)
  - [x] `dpo_loss` / `DpoConfig` (dpo.rs) — DPO loss with per-pair and batch variants; `dpo_log_ratio` helper
  - [x] `ipo_loss` / `IpoConfig` (ipo.rs) — IPO squared loss `(log_ratio_diff - 1/(2β))²`
  - [x] `kto_loss` / `KtoConfig` (kto.rs) — KTO with desirable λ_D and undesirable λ_U; KL reference point z₀=ln2

- [x] **Reference-free alignment** (orpo/)
  - [x] `orpo_loss` / `OrpoConfig` (orpo.rs) — ORPO: `L_SFT + λ·(-log σ(log_odds_w - log_odds_l))`; no reference model
  - [x] `simpo_loss` / `SimpoConfig` (simpo.rs) — SimPO: length-normalized `-log σ(β/|y_w|·Σ lp_w - β/|y_l|·Σ lp_l - γ)`; margin γ

- [x] **RLHF-PPO utilities** (ppo_rlhf/)
  - [x] `RlhfRollout` (rollout.rs) — rollout buffer with log_probs, ref_log_probs, rewards, values; `compute_advantages()` (GAE), `apply_kl_penalty()` (reward -= β·KL)
  - [x] `KlController` (kl_control.rs) — adaptive KL beta: `β *= (1 + k·(kl-target)/target)`; `kl_divergence_from_logps()`
  - [x] `rlhf_ppo_loss` / `RlhfPpoConfig` (ppo_step.rs) — clipped PPO surrogate + value loss + entropy bonus → (policy, value, entropy, approx_kl)

- [x] **SFT loss** (sft/)
  - [x] `sft_loss` / `masked_token_ce` (loss.rs) — cross-entropy with attention mask; logsumexp trick for numerical stability; division by sum of mask

- [x] **Metrics** (metrics/)
  - [x] `win_rate`, `reward_gap`, `kl_from_ref`, `perplexity`, `AlignmentMetrics` (alignment.rs) — standard RLHF evaluation metrics; `compute_alignment_metrics()` batch helper

- [x] **Integration tests** (lib.rs) — 12 e2e tests: BT loss zero for equal rewards, BT decreases with gap, DPO finite, DPO lower for aligned pairs, IPO finite, KTO non-negative, ORPO structure, SimPO length-normalized, SFT correct prediction, KL zero at ref, RewardNormalizer unit variance, PTX kernels × 6 SM versions

- [x] **Benchmarks** (benches/rlhf_ops.rs) — 7 PTX kernel groups × 4 SM versions + 6 algo benches: dpo_batch_256, ipo_batch_256, kto_batch_256, sft_512tokens_32kvocab, reward_norm_update

---

## Vol.33: Meta-Learning Algorithms [COMPLETE]

### oxicuda-meta (27 files, ~4,800 SLoC, 18 tests)

Pure-Rust meta-learning library covering: MAML (Model-Agnostic Meta-Learning with second-order finite-difference gradients), FOMAML (first-order approximation), ANIL (Almost No Inner Loop, head-only adaptation), Reptile (first-order interpolation), Prototypical Networks (class prototype mean + Euclidean distance), Matching Networks (cosine attention over support set), and Relation Networks (2-layer relation MLP). Includes N-way K-shot episode sampling, MLP backbone with Xavier init, and few-shot accuracy metrics with 95% CI.

- [x] **Error types** (error.rs) — `MetaError` (15 variants): DimensionMismatch, EmptySupport, InvalidNWay, InvalidKShot, InvalidFeatDim, InsufficientClasses, InsufficientExamples, InvalidLr, NanEncountered, InvalidQuerySize, InvalidEpisodeConfig, GradientFailure, BackboneError, Internal, InvalidStepSize

- [x] **Handle** (handle.rs) — `SmVersion`, `LcgRng` (Knuth MMIX 64-bit LCG), `MetaHandle::default_handle()` (SM 8.0, device 0, seed 42)

- [x] **PTX kernels** (ptx_kernels.rs) — 7 GPU kernels × 6 SM versions (75/80/86/90/100/120):
  - [x] `inner_sgd_ptx` — `θ'[i] = θ[i] - α·g[i]` elementwise vector SGD step
  - [x] `reptile_update_ptx` — `θ[i] += ε·(θ'[i] - θ[i])` interpolation step
  - [x] `proto_distance_ptx` — squared L2: `d[q,k] = Σ_j (q_j - proto_j)²` for query × prototype pairs
  - [x] `cosine_sim_ptx` — `cos(a,b) = a·b / (||a||·||b|| + ε)` with normalization for MatchingNet
  - [x] `relation_score_ptx` — concat(q,s) + 2-layer ReLU MLP → sigmoid score for RelationNet
  - [x] `meta_grad_accum_ptx` — sum task gradients elementwise, divide by n_tasks
  - [x] `episode_sample_ptx` — LCG-based class/example selection for N-way K-shot episodes

- [x] **Episode utilities** (episode/)
  - [x] `FewShotEpisode` / `EpisodeConfig` (types.rs) — N-way K-shot episode struct; support/query splits with shape validation; `support_for_class()` view helper
  - [x] `EpisodeSampler` (sampler.rs) — LCG Fisher-Yates sampling of N classes then K+Q examples per class from flat dataset

- [x] **Network** (network/)
  - [x] `MlpBackbone` (backbone.rs) — MLP with ReLU (except final linear); Xavier init `U(-√(6/(in+out)), √(6/(in+out)))`; `to_params()` / `from_params()` for MAML parameter flattening
  - [x] `LinearHead` (linear_head.rs) — linear probe classifier with `to_params()` / `from_params()`

- [x] **Gradient utilities** (gradient/)
  - [x] `inner_sgd_step` / `multi_step_inner` / `cross_entropy_loss` (inner_loop.rs) — SGD step + multi-step inner loop via closure; CE loss for classification
  - [x] `fd_gradient` (finite_diff.rs) — central finite differences `(f(θ+ε·eᵢ) - f(θ-ε·eᵢ)) / (2ε)` for gradient approximation

- [x] **MAML family** (maml/)
  - [x] `maml_adapt` / `maml_meta_update` / `MamlConfig` (maml.rs) — MAML: inner adaptation + outer FD meta-gradient; `θ_new = θ - β·∇_θ·Σ_i L(θ'_i)`
  - [x] `fomaml_update` / `FoMamlConfig` (fomaml.rs) — FOMAML: gradient at θ' treated as constant (stop-gradient); cheaper than full MAML
  - [x] `anil_adapt_head` / `anil_meta_update` / `AnilConfig` (anil.rs) — ANIL: only linear head updated in inner loop; body fully shared + frozen

- [x] **Reptile** (reptile/)
  - [x] `reptile_update` / `ReptileConfig` (reptile.rs) — `θ ← θ + ε·(avg_{τ_i}(θ'_i) - θ)`; k inner SGD steps per task

- [x] **Metric learning** (metric_learning/)
  - [x] `compute_prototypes` / `proto_predict` / `proto_loss` (proto_net.rs) — ProtoNet: class prototype = mean(support feats); prediction via argmin L2 to prototypes; CE loss over -d² logits
  - [x] `cosine_similarity` / `matching_net_attention` / `matching_net_predict` (matching_net.rs) — MatchingNet: softmax cosine attention over support; temperature scaling
  - [x] `RelationNet` (relation_net.rs) — 2-layer MLP: concat(q,s) → ReLU → sigmoid; `relation_score`, `predict_episode`, `relation_loss` (MSE on 0/1 targets)

- [x] **Metrics** (metrics/)
  - [x] `episode_accuracy` / `mean_and_ci95` / `accuracy_at_k` (few_shot.rs) — episode accuracy; mean ± 95% CI over episodes; top-k accuracy

- [x] **Integration tests** (lib.rs) — 12 e2e tests: ProtoNet correct class, identity features → correct label, MatchingNet attention sums to 1, same-class highest attention, RelationNet same > different, relation loss finite, MAML adapt changes params, Reptile moves toward task, inner SGD decreases loss, episode sampler correct shapes, 100% accuracy → 1.0, PTX kernels × 6 SM versions

- [x] **Benchmarks** (benches/meta_ops.rs) — 7 PTX kernel groups × 4 SM versions + 6 algo benches: proto_net_5way5shot_d64, matching_net_attention, maml_adapt_inner, reptile_update, episode_sampler

---

## Vol.34: Neural Radiance Fields & Neural Rendering [COMPLETE]

### oxicuda-nerf (17 files, ~6,800 SLoC, 62 tests)

Pure-Rust neural rendering library covering: NeRF positional encoding (sin/cos with L frequency levels, configurable include_input), Instant-NGP multi-resolution hash grid (L levels, T buckets, F features per entry; spatial hashing with primes π2=2654435761, π3=805459861; trilinear interpolation over 8 corners), Mip-NeRF integrated positional encoding (Gaussian attenuation `exp(-ω²σ²/2)` for anti-aliasing), TensoRF CP decomposition (rank-R factored density + color field with 1D axis interpolation), volume rendering (alpha compositing `α_i = 1-exp(-σ_i·δ_i)`, transmittance, early termination T<1e-4), stratified sampling, importance resampling (inverse-CDF), pinhole camera ray generation (c2w 3×4 matrix), occupancy grid acceleration, PSNR/MSE metrics. Zero CUDA SDK dependency.

- [x] **Error types** (error.rs) — `NerfError` (16 variants): DimensionMismatch, EmptyInput, InvalidFreqLevels, InvalidHashConfig, NanEncountered, InvalidBounds, InvalidSampleCount, ZeroRayDirection, InvalidCameraIntrinsics, InvalidGridResolution, HashLevelOutOfRange, InvalidFeatureDim, TensorDecompError, VolumeRenderError, InvalidEncoding, Internal

- [x] **Handle** (handle.rs) — `SmVersion`, `LcgRng` (Knuth MMIX 64-bit LCG), `NerfHandle::default_handle()` (SM 8.0, device 0, seed 42)

- [x] **PTX kernels** (ptx_kernels.rs) — 7 GPU kernels × 6 SM versions (75/80/86/90/100/120):
  - [x] `positional_encoding_ptx` — sin/cos frequency encoding of coordinate batch
  - [x] `volume_render_ptx` — single-ray alpha compositing with T cutoff
  - [x] `hash_grid_lookup_ptx` — multi-res spatial hash + trilinear interpolation
  - [x] `ray_march_ptx` — stratified sample generation along ray
  - [x] `sh_to_rgb_ptx` — SH basis eval for L=0..3 (16 coefficients, view-dependent color)
  - [x] `occupancy_update_ptx` — threshold density → bool occupancy grid
  - [x] `importance_resample_ptx` — inverse-CDF resampling from coarse weight histogram

- [x] **Positional encodings** (encoding/)
  - [x] `positional_encode` / `PosEncConfig` (positional.rs) — `γ(p) = [sin(2^k·π·p), cos(2^k·π·p)]` for k=0..L-1 per dimension; optional raw input concatenation
  - [x] `HashGrid` / `HashGridConfig` (hash_grid.rs) — multi-resolution hash with trilinear lerp; `query()` → `[n_levels * F]`; `query_batch()`
  - [x] `integrated_pe` / `IpeConfig` (integrated_pe.rs) — IPE for Mip-NeRF: `sin(ωμ)·exp(-ω²σ²/2)`, `cos(ωμ)·exp(-ω²σ²/2)`

- [x] **Rendering** (rendering/)
  - [x] `Ray` / `PinholeCamera` (ray.rs) — `Ray::at(t)`, `Ray::normalized()`; camera `ray_through_pixel(u, v, c2w)`, `generate_rays(c2w)`
  - [x] `stratified_sample` / `importance_sample` / `merge_samples` (sampling.rs) — hierarchical NeRF sampling
  - [x] `volume_render` / `volume_render_batch` / `RenderResult` (volume_render.rs) — alpha compositing with depth and opacity output
  - [x] `OccupancyGrid` (occupancy.rs) — resolution³ bool grid; `is_occupied_world()`, `update_from_density()`, `march_ray_occupied()`

- [x] **Networks** (network/)
  - [x] `NerfMlp` / `NerfMlpConfig` (nerf_mlp.rs) — 8-layer ResNet MLP: skip connection at layer 4, sigma head (ReLU), color head (Sigmoid); batch forward
  - [x] `TinyNerf` (tiny_nerf.rs) — compact 4-layer MLP for tests

- [x] **Fields** (field/)
  - [x] `TensorRf` / `TensorRfConfig` (tensorf.rs) — CP decomposition: rank-R factored density (relu) + color; `query_density()`, `query_color()`
  - [x] `HashField` (hash_field.rs) — Instant-NGP style: HashGrid + 2-layer MLP decoder → (sigma, color_feat)

- [x] **Camera** (camera/pinhole.rs) — re-export of PinholeCamera

- [x] **Metrics** (metrics/image_quality.rs) — `psnr()`, `mse_image()`, `compute_image_metrics()` → `ImageMetrics`

- [x] **Integration tests** (lib.rs) — 12 e2e tests: PE shape, deterministic, hash grid shape, trilinear corners distinct, volume render empty=zero, opaque first sample, stratified count, importance count, TensoRF nonneg density, TinyNerf finite, PSNR identity, PTX kernels × 6 SM versions

- [x] **Benchmarks** (benches/nerf_ops.rs) — 7 PTX kernel groups × 4 SM versions + 6 algo benches: pos_enc_1024pts, hash_grid_batch_1024, volume_render_64rays, stratified_sample_128, tensorf_density_1024

---

## Vol.35: Mixture of Experts (MoE) [COMPLETE]

### oxicuda-moe (14 files, ~6,200 SLoC, 60 tests)

Pure-Rust Mixture of Experts primitives covering: Switch Transformer top-1 routing (capacity buffer, overflow token dropping), GShard-style top-K gating (softmax over experts, partial-sort top-k, noise jitter), Expert Choice routing (experts select preferred tokens, guaranteed load balance), Soft MoE (differentiable slot routing D=softmax(X·Φ/√d), slot-aggregated expert inputs), standard GELU/SiLU/ReLU expert FFN with Xavier init, SwiGLU expert (SiLU(W1·x)⊙(W3·x)·W2), ExpertBank + SwiGluBank dispatch, Switch load-balance loss (n_e·Σ f_i·P_i), router z-loss (log²(logsumexp(logits))), routing entropy, expert utilization metrics, full MoeLayer combining all components. Zero CUDA SDK dependency.

- [x] **Error types** (error.rs) — `MoeError` (14 variants): DimensionMismatch, EmptyInput, InvalidExpertCount, InvalidTopK, InvalidCapacityFactor, ExpertIndexOutOfRange, NanEncountered, InvalidHiddenDim, InvalidInputDim, DispatchFailed, RouterNotInitialized, ExpertFfnError, SlotAssignmentError, Internal

- [x] **Handle** (handle.rs) — `SmVersion`, `LcgRng` (Knuth MMIX 64-bit LCG), `MoeHandle::default_handle()` (SM 8.0, device 0, seed 42)

- [x] **PTX kernels** (ptx_kernels.rs) — 7 GPU kernels × 6 SM versions (75/80/86/90/100/120):
  - [x] `top_k_gate_ptx` — softmax + top-k selection per token
  - [x] `expert_dispatch_ptx` — capacity-bounded token→expert slot assignment
  - [x] `expert_ffn_ptx` — batched `y = W2·GeLU(W1·x+b1)+b2` per token
  - [x] `expert_combine_ptx` — weighted sum of expert outputs by gate scores
  - [x] `load_balance_loss_ptx` — `n_e·Σ f_i·P_i` reduction
  - [x] `router_z_loss_ptx` — `log²(logsumexp(logits))` per token, reduction
  - [x] `soft_moe_dispatch_ptx` — slot dispatch D[t,s] = softmax(x·Φ/√d)

- [x] **Routing** (routing/)
  - [x] `TopKRouter` / `TopKConfig` / `TopKResult` / `topk()` (top_k.rs) — k=1 argmax, k=2 one-pass max2, k≥3 partial sort; optional noise jitter `N(0, σ²)` via Box-Muller; top-k score normalization
  - [x] `switch_dispatch` / `switch_combine` / `SwitchDispatch` (switch.rs) — capacity = ceil(T/E·cap_factor), overflow tracking, combine via gate scores
  - [x] `expert_choice_route` / `expert_choice_combine` / `ExpertChoiceResult` (expert_choice.rs) — experts select top-c tokens from their score column; guaranteed load balance
  - [x] `SoftMoeRouter` (soft_moe.rs) — `dispatch_weights()` D=[T,E·S], `aggregate_inputs()` (weighted avg per slot), `combine_outputs()` (scatter back to tokens)

- [x] **Expert FFN** (expert/)
  - [x] `ExpertFfn` / `ExpertActivation` / `SwiGluExpert` (ffn.rs) — GELU/SiLU/ReLU activation; SwiGLU `(SiLU(W1·x)⊙W3·x)·W2`; Xavier init; batch forward
  - [x] `ExpertBank` / `SwiGluBank` (bank.rs) — N-expert bank; `forward_expert(idx, tokens)`; `forward_dispatched(x, assignments, scores)`

- [x] **Losses** (loss/)
  - [x] `load_balance_loss` / `compute_load_stats` / `LoadStats` (load_balance.rs) — `L_aux = n_e·Σ f_i·P_i`; per-expert fraction and mean probability
  - [x] `router_z_loss` (router_z.rs) — `(1/B)·Σ_b log²(LSE_b)` with stable logsumexp
  - [x] `routing_entropy` (entropy.rs) — `-(1/T)·Σ_t Σ_e p_{t,e}·log(p_{t,e}+ε)`

- [x] **Metrics** (metrics/utilization.rs) — `ExpertUtilization { tokens_per_expert, overflow_count, load_imbalance_ratio, utilization_fraction }` via `compute_utilization()`

- [x] **MoeLayer** (layer/moe_layer.rs) — `MoeLayer { router, experts }`: full forward (route → Switch dispatch → expert bank → combine → aux losses); `MoeLayerOutput { hidden, aux_loss, n_overflows, load_stats }`

- [x] **Integration tests** (lib.rs) — 12 e2e tests: top-k scores sum to 1, indices valid, Switch capacity respected, overflow counted, ExpertFfn finite, output shape, SwiGLU finite, load balance nonneg, z-loss nonneg, Soft MoE dispatch sums to 1, MoeLayer forward shape, PTX kernels × 6 SM versions

- [x] **Benchmarks** (benches/moe_ops.rs) — 7 PTX kernel groups × 4 SM versions + 6 algo benches: topk_routing_512tok_8exp_d256, expert_ffn_batch64, switch_dispatch_512tok, load_balance_512tok, moe_layer_128tok

---

## Vol.36: Tabular Deep Learning [COMPLETE]

`oxicuda-tabular` — Pure-Rust tabular deep learning primitives: sparse probability transforms, attention-based networks, transformers, and neural decision trees.

- [x] **attention/sparsemax.rs** — `sparsemax` (Martins & Astudillo 2016): sort-descending k* algorithm, O(d log d). `entmax15` (α=1.5): bisection 64 iterations. `sparsemax_batch` for batched row-wise projection.
- [x] **attention/tabnet.rs** — TabNet (Arik & Pfister 2021): `glu` (GLU gate), `BatchNorm1d` (γ/β learnable), `TabNetConfig` {n_features, n_d, n_a, n_steps, gamma, n_classes}, `TabNetLayer` (Xavier init, step-wise sparsemax attention, prior scales P_i=Π(γ-M_j), shared + step-specific FC-BN-GLU blocks).
- [x] **attention/saint.rs** — SAINT (Somepalli et al. 2021): `self_attention` (scaled dot-product), `multihead_attention`, `intersample_attention` (per-feature cross-sample), `SaintConfig`, `SaintLayer` (alternating row-wise + intersample MHSA with Pre-LN FFN, CLS mean-pool head).
- [x] **transformer/ft_transformer.rs** — FT-Transformer (Gorishniy et al. 2021): `FeatureTokenizer` (continuous: x_j·w_j+b_j per embed dim; categorical: lookup table), `FtConfig`, `FtTransformer` (Pre-LN MHSA blocks, CLS token, linear head).
- [x] **tree/node.rs** — NODE (Popov et al. 2019): `NodeConfig`, `NodeTree` (depth-d soft oblivious decision tree, entmax-1.5 feature selection, sigmoid-smoothed splits, leaf tensor products), `NodeEnsemble` (mean over trees).
- [x] **preprocess/normalize.rs** — `QuantileNormalizer` (empirical rank [0,1], binary-search transform), `StandardNormalizer` (z-score, Welford-style std), `MinMaxNormalizer`.
- [x] **preprocess/embed.rs** — `FeatureEmbedder` (fit continuous μ/σ, validate categorical ranges).
- [x] **metrics/tabular_metrics.rs** — `binary_accuracy`, `multiclass_accuracy` (argmax), `rmse`, `mae`, `auc_roc` (sort-by-score + trapezoidal), `compute_binary_metrics`, `ClassificationMetrics`.
- [x] **PTX kernels** (ptx_kernels.rs) — 7 kernels × 6 SM versions (75/80/86/90/100/120): `sparsemax_kernel` (per-row simplex projection), `feature_tokenize_kernel` (FT-Transformer continuous tokenisation), `tabnet_step_attn_kernel` (prior-scaled QK dot + sparsemax), `intersample_attn_kernel` (SAINT cross-sample QKᵀ/√d), `node_tree_eval_kernel` (soft oblivious routing + leaf weighting), `quantile_norm_kernel` (binary-search rank), `auc_roc_kernel` (sorted-label trapezoidal accumulation).
- [x] **Integration tests** (lib.rs) — 12 e2e tests: sparsemax sums to 1, entmax sums to 1, TabNet output shape + attention non-negative, SAINT forward shape, FT-Transformer finite logits, NODE ensemble shape, QuantileNormalizer range [0,1], binary accuracy perfect, AUC-ROC perfect, PTX kernels × 6 SM versions, FT-Transformer batch.
- [x] **Benchmarks** (benches/tabular_ops.rs) — 7 PTX kernel groups × 4 SM versions + 5 algo benches.
- **Tests**: 51 passing

---

## Vol.37: Anomaly Detection [COMPLETE]

`oxicuda-anomaly` — Pure-Rust anomaly detection library covering deep learning, distance-based, density-based, statistical, and ensemble methods.

- [x] **svdd/deep_svdd.rs** — DeepSVDD (Ruff et al. 2018): 3-layer MLP, no bias in last layer (hypersphere collapse prevention), `fit` computes fixed center c=mean(φ(x_i)) (adjusted if near-zero), `score` = ||φ(x)-c||², Xavier init.
- [x] **reconstruction/autoencoder.rs** — `AeConfig`, `AnomalyAutoencoder` (Xavier init, ReLU encoder + sigmoid decoder), `score` = MSE reconstruction error, batch scoring.
- [x] **reconstruction/vae_anomaly.rs** — `VaeConfig`, `AnomalyVae` (μ/log_var encoder, Box-Muller reparametrize, MSE+KL β-ELBO), deterministic-μ scoring (no stochasticity at inference).
- [x] **distance/lof.rs** — LOF (Breunig et al. 2000): brute-force k-NN, `fit` computes knn_indices/knn_dists/lrd; `reach_dist_k(i,j)=max(knn_dists[j*(k-1)], dist(i,j))`; `lrd_k(i)=k/Σreach_dist`; `score=mean(lrd_neighbors)/lrd_x`. Numerical guard: lrd → 1e30 if zero denominator.
- [x] **distance/knn_score.rs** — `KnnScorer`: average k-NN distance baseline, brute-force, batch scoring.
- [x] **density/copod.rs** — COPOD (Li et al. 2020): empirical CDF via sorted column binary search; `score=-Σ(log(F_j(x_j))+log(1-F_j(x_j)))/2`; `score_skew_adjusted` (Fisher-Pearson skewness weighted tail).
- [x] **density/mahalanobis.rs** — `MahalanobisDetector`: sample mean + covariance estimation; Gauss-Jordan inversion with full pivoting on augmented [M|I]; ridge 0.01·I for numerical stability; D²=diffᵀ·Σ⁻¹·diff.
- [x] **isolation/iforest_score.rs** — `IsolationScorer`: random projection path-length estimation; `c_factor(n)=2H(n-1)-2(n-1)/n` (EULER_MASCHERONI=0.5772156649); `isolation_score_from_path(avg_path,n)=2^(-avg_path/c_n)`.
- [x] **statistical/stats.rs** — `MadDetector` (MAD=1.4826·median|xi-μ|, Z-score via MAD, configurable threshold), `ZScoreDetector` (Welford online μ/σ, |x-μ|/σ>threshold), `percentile_threshold` (linear interpolation).
- [x] **ensemble/ensemble.rs** — `AnomalyEnsemble` (Average/Maximum/Weighted combiners with per-detector min-max normalisation; `add_detector`, `score_ensemble`).
- [x] **metrics/anomaly_metrics.rs** — `auc_roc`, `auc_pr` (precision-recall trapezoidal), `f1_at_threshold`, `compute_detection_metrics` (AUC-ROC + AUC-PR + F1 together).
- [x] **PTX kernels** (ptx_kernels.rs) — 7 kernels × 6 SM versions (75/80/86/90/100/120): `svdd_loss_kernel` (||z-c||² per sample), `recon_score_kernel` (MSE per sample via warp-shuffle), `lof_reach_dist_kernel` (k-dist lookup + max), `copod_ecdf_kernel` (binary search rank), `mahal_dist_kernel` (quadratic form per sample), `iforest_score_kernel` (2^(-avg_path/c_n) via ex2.approx), `ensemble_normalize_kernel` (per-detector min-max then mean).
- [x] **Integration tests** (lib.rs) — 12 e2e tests: DeepSVDD self score < mean, AE recon score finite, VAE score finite, LOF self score ≤ mean neighbor score, kNN self score 0, COPOD score finite, Mahalanobis self score 0, IsolationScorer anomaly > typical, MAD score vs threshold, ZScore flag outlier, Ensemble normalised sum, PTX kernels × 6 SM versions.
- [x] **Benchmarks** (benches/anomaly_ops.rs) — 7 PTX kernel groups × 4 SM + 5 algo benches.
- **Tests**: 62 passing

---

## Vol.38: Quantum Simulation [COMPLETE]

Pure-Rust quantum simulation and QML library covering state-vector simulation, standard and parametric gates (H/S/T/Rx/Ry/Rz/U3/CNOT/CZ/SWAP/CCX), Pauli strings, Hamiltonians, and expectation values; Trotter-Suzuki 1st/2nd/4th-order evolution and Lindblad master-equation density-matrix stepping; VQE with hardware-efficient ansatz and parameter-shift gradient descent; QAOA for Max-Cut/Ising; density matrices with partial-trace and quantum-information metrics (purity/fidelity/von-Neumann entropy); Kraus channels (depolarizing/amplitude-damping/phase-damping); angle/amplitude/ZZ-feature-map embeddings; overlap quantum kernel (QML); QuantumCircuit DSL.

### oxicuda-quantum (~34 files, ~6,500 SLoC)
- [x] **statevec** — `StateVector`, `apply_1q_inplace` (bit-mask), `apply_2q_inplace` (4×4 complex matmul), `apply_1q_controlled`
- [x] **gates** — `gate_{i,x,y,z}`, `gate_{h,s,t,sdg,tdg}`, `apply_{cnot,cz,swap,ccx}`, `gate_{rx,ry,rz,u3,phase}`
- [x] **pauli** — `PauliOp`, `PauliString`, `Hamiltonian`, `expectation_value` (basis-rotation + parity counting)
- [x] **trotter** — `TrotterStep` (1st/2nd/4th-order Suzuki-Yoshida), `LindbladOp`, `lindblad_step`
- [x] **vqe** — `HardwareEfficientAnsatz`, `VqeOptimizer` (parameter-shift gradient + SGD)
- [x] **qaoa** — `QaoaCircuit` (cost+mixer p-layer alternation, energy via cut evaluation)
- [x] **density** — `DensityMatrix`, `partial_trace` (index folding), `purity`, `fidelity`, `von_neumann_entropy`
- [x] **channel** — `KrausChannel` (completeness-checked), `depolarizing_channel`, `amplitude_damping_channel`, `phase_damping_channel`
- [x] **embedding** — `angle_embedding`, `amplitude_embedding`, `zz_feature_map` (Havlíček depth-2)
- [x] **kernel** — `overlap_kernel` K(x,y)=|⟨ψ(x)|ψ(y)⟩|², `kernel_matrix`
- [x] **circuit** — `QuantumCircuit`, `GateOp` enum, `exec_on_state`
- [x] **PTX kernels** (ptx_kernels.rs) — 7 kernels × 6 SM versions: `statevec_apply_1q`, `statevec_apply_2q`, `statevec_apply_cnot`, `expval_pauli`, `partial_trace`, `trotter_step`, `measure_prob`
- [x] **Integration tests** (lib.rs) — 14 e2e tests: |0⟩ norm, H superposition, HH=I, CNOT Bell state, Pauli-Z ⟨0|, Pauli-Z ⟨1|, mixed Hamiltonian, VQE converges, QAOA runs, density purity=1, depolarizing reduces purity, PTX all SM versions
- [x] **Benchmarks** — 7 PTX kernel groups × 4 SM + 5 algo benches
- **Tests**: 61 passing

---

## Vol.39: Approximate Nearest Neighbor & Vector Search [COMPLETE]

Pure-Rust ANN and vector-search library covering brute-force baseline, top-K heap, mini-batch k-means++, Product Quantization (PQ) with asymmetric distance computation (ADC), IVF coarse quantizer, IVF+PQ (IVFPQ), HNSW (Malkov & Yashunin 2018 with neighbor-selection heuristic), locality-sensitive hashing (random-projection cosine, MinHash, SimHash), NN-Descent k-NN graph build, and scalar quantizers (SQ8, SQ4).

### oxicuda-ann (~32 files, ~6,200 SLoC)
- [x] **distance** — `l2_sq`, `l2`, `l2_sq_all` (batch), `ip`, `cosine_sim`, `hamming_u32`, `hamming_f32_packed`
- [x] **flat** — `FlatIndex`, `search_l2`, `search_ip` (brute-force baseline, min-heap top-K)
- [x] **topk** — `BoundedMaxHeap` (k-smallest via max-heap eviction), `select_topk`
- [x] **kmeans** — `KMeans`, `kmeans_pp_init`, mini-batch fit (25 epochs default)
- [x] **pq** — `PqCodebook`, `train_pq` (per-subspace k-means), `encode_vector/batch`, `build_adc_table`, `adc_distance`; `OpqModel` (OPQ alternating Procrustes rotation, Ge et al. 2013); `AnisotropicPq` (ScaNN-style query-direction weighted k-means, Guo et al. 2020)
- [x] **ivf** — `IvfIndex` (coarse k-means quantizer + posting lists), `search` (top-nprobe)
- [x] **ivfpq** — `IvfPq` (coarse-prune + ADC search within top-nprobe lists)
- [x] **hnsw** — `HnswGraph`, `hnsw_insert` (level draw + select_neighbors_heuristic), `hnsw_search` (greedy descent + ef-bounded BFS)
- [x] **lsh** — `RandomProjLsh` (sign-bit hash), `MinHash` (Jaccard via LCG hash families), `SimHash` (cosine sim via Gaussian projections)
- [x] **knn_graph** — `KnnGraph`, `build_brute` (O(n²) baseline), `build_nn_descent` (NN-Descent iteration)
- [x] **quantize** — `Sq8Quantizer` (per-dim min/max → uint8), `Sq4Quantizer` (4-bit nibble packed)
- [x] **PTX kernels** (ptx_kernels.rs) — 7 kernels × 6 SM versions: `l2_distance_batch`, `ip_distance_batch`, `pq_adc_table`, `hnsw_neighbor_eval`, `ivf_assign`, `lsh_random_proj`, `topk_select`
- [x] **Integration tests** (lib.rs) — 12 e2e tests: FlatIndex exact match, top-K count, k-means clusters, PQ train, PQ ADC distance, IVF search, HNSW self-find, HNSW recall ≥80%, LSH self-hash, MinHash Jaccard, NN-Descent quality, PTX all SM versions
- [x] **Benchmarks** — 7 PTX kernel groups × 4 SM + 5 algo benches
- **Tests**: 32 passing

---

## Vol.40: Recommender Systems [COMPLETE]

Pure-Rust recommender-system library covering classical matrix factorization (ALS implicit-feedback, BPR, NMF), neural CF (NCF GMF⊕MLP, Two-Tower DSSM), feature-crossing models (DeepFM second-order FM, AutoInt multi-head self-attention, Wide&Deep), sequence-aware models (GRU4Rec, SASRec causal self-attention, BERT4Rec bidirectional MLM), graph-based recommenders (LightGCN symmetric-normalized propagation, NGCF interaction-aware aggregation), multi-task learners (MMoE shared experts, PLE cascaded experts, ESMM pCTR×pCVR), negative sampling (uniform rejection, popularity-biased CDF, hard-negative mining), and ranking metrics (Precision@K, Recall@K, NDCG@K, MAP@K, MRR, HitRate@K, AUC Wilcoxon-Mann-Whitney).

### oxicuda-recsys (~34 files, ~6,800 SLoC)
- [x] **factorization** — `Als` (Gauss-Jordan closed-form, implicit c_ui=1+α·r), `Bpr` (triplet gradient σ(x_ui−x_uj)), `Nmf` (multiplicative update W/H rules)
- [x] **ncf** — `Ncf` (GMF element-wise product ⊕ MLP concat → sigmoid)
- [x] **two_tower** — `TwoTower` (dual MLP user/item encoders + dot-product score)
- [x] **deepfm** — `DeepFm` (linear + FM 2nd-order 0.5·((Σe)²−Σe²) + Deep MLP), `AutoInt` (multi-head self-attention over field embeddings + residual), `WideDeep` (linear ⊕ MLP)
- [x] **sequential** — `Gru4Rec` (full GRU cell z/r/n gates), `SasRec` (causal self-attention + FFN + LayerNorm), `Bert4Rec` (bidirectional attention + MLM masking)
- [x] **graph_recsys** — `LightGcn` (D⁻½AD⁻½ propagation + layer-mean pooling), `Ngcf` (LeakyReLU aggregation + concat layers)
- [x] **multitask** — `Mmoe` (per-task softmax gates over shared experts), `Ple` (cascaded shared+task-specific expert layers), `Esmm` (pCTR × pCVR = pCTCVR product)
- [x] **sampling** — `UniformNegSampler` (rejection, max 100 tries), `PopularityNegSampler` (CDF binary search), `HardNegSampler` (top-20% non-positive pool)
- [x] **metrics** — `precision_at_k`, `recall_at_k`, `ndcg_at_k` (DCG/IDCG log2), `map_at_k`, `mrr`, `hit_rate_at_k`, `auc_recsys` (Wilcoxon-Mann-Whitney with tie handling)
- [x] **PTX kernels** (ptx_kernels.rs) — 7 kernels × 6 SM versions: `als_step_ptx` (Cholesky-style), `bpr_grad_ptx`, `embedding_lookup_ptx`, `dot_score_ptx`, `softmax_topk_ptx`, `negsample_uniform_ptx`, `lightgcn_propagate_ptx`
- [x] **Integration tests** (lib.rs) — 12 e2e tests: ALS score finite, BPR loss finite, NMF fit, NCF in [0,1], TwoTower score finite, DeepFM in [0,1], WideDeep in [0,1], SASRec logits finite, LightGCN score finite, NDCG perfect ranking, uniform neg not in positives, PTX non-empty all SM versions
- [x] **Benchmarks** — PTX bench group + NDCG bench + RNG bench
- **Tests**: 12 passing

---

## Vol.41: Causal Inference [COMPLETE]

Pure-Rust causal-inference library covering DAG representation (cycle-safe add/remove, ancestors/descendants/d-separation via Bayes-ball), causal discovery (NOTEARS augmented-Lagrangian linear SEM, PC algorithm Fisher-Z skeleton + Meek orientation rules, Greedy Equivalence Search BIC-scored, NOTEARS-MLP column-norm acyclicity), treatment-effect estimation (propensity logistic GD, IPW ATE/ATT, S/T/X-learners over OLS base, AIPW doubly robust, Chernozhukov DML K-fold cross-fitting, DragonNet shared representation + 3 heads + targeted regularization), instrumental variables (2SLS two-stage OLS, DeepIV two-stage MLP), causal forests (Wager-Athey 2018 honest splitting + heterogeneous split criterion), twin-network counterfactuals (shared encoder + dual decoder), do-calculus (backdoor/frontdoor criterion evaluation, G_x_bar d-separation check), and causal metrics (PEHE, ATE bias, policy risk, qini coefficient, R²_CATE).

### oxicuda-causal (~31 files, ~6,400 SLoC)
- [x] **dag** — `Dag` (adjacency matrix, cycle-safe BFS add_edge, Kahn topo_sort), `d_separated` (Bayes-ball algorithm with collider handling)
- [x] **discovery** — `NotearsSem` (augmented Lagrangian + Padé(3,3) matrix-exponential acyclicity h(W)=tr(e^{W⊙W})−d), `PcAlgorithm` (Fisher-Z skeleton + Meek rules R1–R4 orientation), `Ges` (BIC-score forward/backward greedy), `NotearsNlp` (MLP first-layer column-norm acyclicity)
- [x] **effect** — `PropensityModel` (logistic GD sigmoid), `ipw_ate`/`ipw_att` (propensity clipped [0.05,0.95]), `SLearner`/`TLearner`/`XLearner` (OLS base), `aipw_ate` (AIPW doubly robust), `DoubleML` (K-fold cross-fitted residuals θ̂=mean[(T-m̂)(Y-ĝ)]/mean[(T-m̂)²]), `DragonNet` (shared repr + μ₀,μ₁,π heads + ε targeted reg)
- [x] **iv** — `TwoSls` (stage-1 OLS T~Z, stage-2 OLS Y~T̂ projection), `DeepIv` (two-stage MLP with ReLU hidden layers)
- [x] **forest** — `CausalForest` (honest estimation: separate build/estimate samples, heterogeneous split score (τ_L−τ_R)²·n_L·n_R/n, random feature subsets √p)
- [x] **counterfactual** — `TwinNetwork` (shared MLP encoder + dual decoder for factual/counterfactual reconstruction)
- [x] **do_calculus** — `backdoor_admissible` (G_x_bar mutilation = remove outgoing edges from X, d-sep check), `frontdoor_admissible`, `backdoor_adjustment` (minimal valid set search)
- [x] **metrics** — `pehe` (√MSE of CATE), `ate_bias`, `policy_risk`, `qini_coeff` (uplift curve area), `r_squared_cate`
- [x] **PTX kernels** (ptx_kernels.rs) — 7 kernels × 6 SM versions: `partial_corr_ptx`, `notears_loss_ptx`, `expm_pade_ptx`, `propensity_logit_ptx`, `ipw_estimator_ptx`, `dml_residual_ptx`, `causal_split_score_ptx`
- [x] **Integration tests** (lib.rs) — 12 e2e tests: DAG add/remove, cycle detection, d-separation chain, PC algorithm, NOTEARS fit acyclic, propensity in [0,1], IPW finite, double-ML finite, causal forest fit/predict, backdoor admissible chain, PTX non-empty all SM versions
- [x] **Benchmarks** — PTX bench group + partial corr bench + DML residual bench
- **Tests**: 43 passing

---

## Vol.42: Parameter-Efficient Fine-Tuning [COMPLETE]

Pure-Rust PEFT library covering the full spectrum of parameter-efficient adaptation methods: LoRA (low-rank adaptation with configurable r/α, Kaiming-uniform A, zero B), QLoRA (NF4 dequantization with 16-bucket lookup table, double-quantization absmax), AdaLoRA (SVD-parameterized ΔW=P·diag(Λ)·Q with importance-score-based rank pruning), DoRA (weight-decomposed magnitude+direction fine-tuning), IA³ (learned per-position scale vectors for K/V/FFN placements), Prefix-Tuning (per-layer K/V prefix reparameterized via MLP), P-Tuning v2 (independent prefix per transformer layer), Prompt-Tuning (soft prompt embeddings prepended to input), Houlsby adapters (dual placement post-attention+FFN with LN, GELU, zero-init up), Pfeiffer adapters (post-FFN only, skip-init), Parallel adapters (FFN-parallel with summation), Compacter (PHM Kronecker low-rank), BitFit (bias-only training identification), Diff-Pruning (concrete distribution L0 relaxation), and LoRA merging utilities (TIES-style sign consensus, DARE random pruning).

### oxicuda-peft (~22 files, ~2,977 SLoC)
- [x] **lora** — `LoraConfig {r, alpha, init_scale}`, `LoraLinear {W, A∈ℝ^{d×r}, B∈ℝ^{r×k}, scale=α/r}`, `merge_into_w`/`unmerge_from_w`, `lora_delta`
- [x] **qlora** — `NF4_TABLE: [f32; 16]` (Dettmers 2023 quantiles), `nf4_quantize/dequantize`, `quantize_block`, `QloraLinear` with double-quant absmax
- [x] **adalora** — `AdaloraLinear {P, Λ, Q}`, `importance_scores = |λ_i|·||P_i||·||Q_i||`, `prune_to_target` (zero Λ below budget), `reconstruct_delta`
- [x] **dora** — `DoraLinear {magnitude, direction_w, A, B}`, column-wise magnitude normalization + direction update
- [x] **ia3** — `Ia3Placement {Key, Value, FeedForward}`, `Ia3Vector {scale}`, element-wise apply `y = x ⊙ scale`
- [x] **prefix** — `PrefixConfig {num_virtual_tokens, prefix_dim, num_layers, num_heads, head_dim}`, `PrefixModule {K_prefix, V_prefix}` per-layer ~N(0,0.02)
- [x] **p_tuning_v2** — `PTuningV2 {layers: Vec<PrefixModule>}` — independent prefix per transformer layer
- [x] **prompt_tuning** — `SoftPrompt {embeddings}`, `prepend_to_sequence` → output len = num_tokens + seq_len
- [x] **adapter/houlsby** — `HoulsbyAdapter {down, up, layer_norm}`, GELU bottleneck, zero-init up, residual
- [x] **adapter/pfeiffer** — post-FFN only, skip-init up projection
- [x] **adapter/parallel** — `ParallelAdapter`, FFN-parallel branch summed at output
- [x] **adapter/compacter** — PHM Kronecker decomposition: ΔW = Σ_i A_i ⊗ B_i
- [x] **bitfit** — `BitFitLayerInfo`, `BitFitMask::for_transformer`, `total_trainable_params`, `is_bias_param`
- [x] **diff_pruning** — `DiffPruner {log_alpha, delta}`, concrete distribution `s=σ((log_α-log(u/(1-u)))/β)` with stretch [γ,ζ], L0 regularizer
- [x] **merge** — `merge_loras` (weighted delta sum), `linear_merge`, `ties_merge` (magnitude prune + majority-vote sign)
- [x] **arithmetic** — `dare_prune` (random density pruning, 1/density rescale), `sign_consensus`, `weighted_sum`
- [x] **metrics/efficiency** — `param_efficiency_ratio`, `effective_rank` (energy-based), `lora_param_count`, `compression_ratio`
- [x] **metrics/merge_test** — `output_mse`, `output_consistency`, `max_abs_diff`
- [x] **PTX kernels** (ptx_kernels.rs) — 7 kernels × 6 SM versions: `lora_matmul_ptx`, `ia3_scale_ptx`, `prefix_expand_ptx`, `adapter_forward_ptx`, `nf4_dequant_ptx`, `lora_merge_ptx`, `prompt_concat_ptx`
- [x] **Integration tests** (lib.rs) — 12 e2e tests: zero-B no-change, scale=α/r, merge-unmerge roundtrip, NF4 range, AdaLoRA importance≥0, prune reduces rank, IA³ identity scale, prefix shape, soft-prompt length, Houlsby residual-init, BitFit count, PTX × 6 SM
- [x] **Benchmarks** (benches/peft_ops.rs) — PTX bench group (lora_matmul, nf4_dequant × 4 SM) + LoRA forward algo bench
- **Tests**: 17 passing

---

## Vol.43: Knowledge Distillation [COMPLETE]

Pure-Rust knowledge distillation library covering the full distillation taxonomy: logit-based (Hinton KD with T²·KL soft-label + CE hard-label, DIST inter/intra-class Pearson correlation, DKD decoupled TCKD+NCKD), feature-based (FitNets linear regressor hint MSE, AT spatial power-sum pooling + L2 normalization, PKT pairwise cosine affinity matrix + KL), relational (RKD pairwise μ-normalized distance loss + angle loss over sampled triplets, CRD InfoNCE with EMA memory bank, CC Gram matrix Frobenius), attention transfer (per-head/per-layer MSE, MiniLM VV^T distillation, MHA with optional 1D Wasserstein), online/mutual (DML N-peer KL aggregation, BYOT auxiliary branch self-distillation from deepest teacher, EMA mean-teacher self-distillation), born-again/iterative (BAN generation ensemble, TAS capacity-gap heuristic + geometric-mean assistant sizing, progressive consistency loss with step-halving), data-free (DAFL MLP generator with teacher confidence + entropy + activation losses, ZSKD Dirichlet class impressions), and distillation metrics (top-K agreement, Cohen's kappa, KL/JS/Wasserstein-1D divergences, param/FLOPs/latency compression ratios).

### oxicuda-distill (~26 files, ~3,685 SLoC)
- [x] **logit/hinton_kd** — `HintonKdConfig {temperature, alpha}`, `softmax_with_temp`, `kl_divergence`, `cross_entropy`, `kd_loss = α·T²·KL + (1-α)·CE`, `kd_loss_batch`
- [x] **logit/dist_distill** — `pearson_corr`, `inter_class_loss`, `intra_class_loss`, `dist_loss(β, γ)`
- [x] **logit/decoupled_kd** — `tckd_loss = -p_t^t·log(p_t^s)`, `nckd_loss = T²·KL(non-target)`, `dkd_loss = α·TCKD + β·NCKD`
- [x] **feature/fitnets** — `FitNetsRegressor {w, b}` (linear, He init), `hint_loss = MSE(proj(s), t)`, `mse`
- [x] **feature/at** — `at_map = Σ_c|F_c|^p`, `l2_normalize`, `at_loss = ||q_s - q_t||²`, `at_loss_batch`
- [x] **feature/pkt** — `cosine_similarity`, `build_affinity_matrix` (row-wise softmax cosine Gram), `pkt_loss = KL(K_t || K_s)`
- [x] **relation/rkd** — pairwise distances, μ-normalization, `smooth_l1`, `distance_loss` (upper-triangle), `angle_loss` (500 random triplets), `rkd_loss = λ_d·dist + λ_a·angle`
- [x] **relation/crd** — `CrdMemoryBank` (EMA momentum update), `crd_loss` (InfoNCE cosine pos/neg)
- [x] **relation/cc** — `gram_matrix = F^T·F / n`, `frobenius_norm_sq`, `cc_loss`
- [x] **attention/attn_distill** — `attn_loss` (MSE), `multi_head_attn_loss`, `multi_layer_attn_loss`
- [x] **attention/value_distill** — `value_relation_matrix = softmax(VV^T)`, `value_relation_loss` (MSE)
- [x] **attention/mha_distill** — `head_attn_mse`, `wasserstein_1d` (CDF difference), `mha_distill_loss` (switchable)
- [x] **online/dml** — `dml_peer_loss = CE + mean_peers KL(self||peer)`, `dml_all_losses` (N-peer all pairs)
- [x] **online/byot** — `BranchClassifier` linear head, `byot_loss` (branch vs deepest teacher KD), `byot_ensemble` (mean logits)
- [x] **online/sd_ema** — `EmaTeacher {params, momentum}`, EMA update `θ_t ← m·θ_t + (1-m)·θ_s`, `ema_loss = α·T²·KL + (1-α)·CE`
- [x] **born_again/ban** — `BanGeneration {generation, params}`, `ban_loss` (KD from gen-k to gen-k+1), `ensemble_logits` (mean)
- [x] **born_again/tas** — `CapacityGap {ratio}`, `needs_assistant (>10×)`, `optimal_assistant_size = √(teacher·student)`, `tas_loss`
- [x] **born_again/progressive** — `ProgressiveConfig {initial_steps, current_steps}`, `next_generation` (halve steps), `consistency_loss` (MSE trajectory), `progressive_distill_step`
- [x] **data_free/dafl** — `DaflGenerator {w1,b1,w2,b2}` (He init, ReLU), `dafl_teacher_loss`, `dafl_info_entropy_loss`, `dafl_activation_loss`, `dafl_total_generator_loss`
- [x] **data_free/zskd** — `dirichlet_sample` (exponential approx + normalize), `class_impression_loss` (CE with Dirichlet target), `synthesize_impression`, `zskd_student_loss`
- [x] **metrics/agreement** — `top_k_agreement`, `cohen_kappa`, `prediction_overlap`
- [x] **metrics/divergence** — `kl_divergence`, `js_divergence = 0.5·KL(p||m)+0.5·KL(q||m)`, `wasserstein_1d` (sorted L1)
- [x] **metrics/compression** — `param_ratio`, `flops_ratio`, `latency_speedup`, `estimate_lora_flops`
- [x] **PTX kernels** (ptx_kernels.rs) — 7 kernels × 6 SM versions: `kd_loss_ptx`, `mse_distill_ptx`, `attn_distill_ptx`, `at_pool_ptx`, `dml_loss_ptx`, `crd_score_ptx`, `gram_matrix_ptx`
- [x] **Integration tests** (lib.rs) — 12 e2e tests: Hinton T=1→CE, identical logits→KL=0, Pearson in [-1,1], AT map shape, AT normalized norm≈1, RKD dist≥0, CRD bank update, CC Gram symmetric, DML count, EMA interpolation, perfect agreement, PTX × 6 SM
- [x] **Benchmarks** (benches/distill_ops.rs) — PTX bench group (kd_loss, gram_matrix × 4 SM) + Hinton KD algo bench
- **Tests**: 76 passing

---

## Vol.44: Optimal Transport [COMPLETE]

Pure-Rust canonical Optimal Transport library covering entropic OT (log-domain Sinkhorn-Knopp with stabilised LSE, debiased Sinkhorn divergence, low-level half-iteration steps), exact OT (network simplex with Northwest-corner-rule basis + Bland's-rule pivots + DFS cycle detection, EMD-1D via sorted breakpoint sweep, EMD dispatch), Wasserstein distances (W1 1D + multi-dim via simplex, W2 1D quantile + multi-dim, sliced-Wasserstein with random unit directions, max-sliced via gradient-ascent on the sphere), Gromov-Wasserstein (entropic GW with `G = -2·C₁·T·C₂^T` outer-loop + Sinkhorn inner, fused-GW combining intra-domain GW + inter-domain Wasserstein with α weight), KL-relaxed unbalanced OT (generalised log-domain Sinkhorn with τ exponent), Wasserstein barycenters (free-support alternating Sinkhorn + barycentric support update, fixed-support Cuturi-Doucet weight update), JKO proximal scheme (heat-equation step + arbitrary-potential variant), Schrödinger Bridge / IPF (log-domain), multi-marginal OT (log-domain tensor scaling alternating projections, k=2 reduces to Sinkhorn), Wasserstein k-means (W2 assignment + barycenter centroid update), OT-based domain adaptation (barycentric mapping `Tx_i = Σ_j (P_ij/Σ_k P_ik)·y_j`), and diagnostic metrics (marginal violation, KL/JS divergences, transport cost, plan entropy).

### oxicuda-ot (~36 files, ~7,765 SLoC)
- [x] **sinkhorn/sinkhorn** — `SinkhornConfig {eps, max_iter, tol}`, `SinkhornResult {plan, u, v, cost, iters}`, log-domain stabilised iterative Bregman projection with row-LSE/col-LSE updates and column-residual convergence detection
- [x] **sinkhorn/divergence** — `sinkhorn_divergence` returns `OT_ε(a,b) − ½(OT_ε(a,a) + OT_ε(b,b))` (Feydy 2019)
- [x] **sinkhorn/log_sinkhorn** — `log_sinkhorn_step_row`, `log_sinkhorn_step_col`, `log_to_plan` low-level half-iteration primitives
- [x] **exact/network_simplex** — `NsConfig {max_iter}`, NW-corner basis, dual potentials over spanning tree, Bland's-rule pivot, DFS cycle detection, mass-shift along θ⁻ legs
- [x] **exact/emd** — `emd_1d` via sorted breakpoint sweep computing `∫|F_a − F_b| dt`, `emd` generic dispatch to network-simplex
- [x] **wasserstein/w1** — `w1_1d` (delegates to `emd_1d`), `w1` multi-dim with L₂ cost via simplex
- [x] **wasserstein/w2** — `w2_1d` quantile sweep, `w2` multi-dim with `½‖·‖²` cost (returns `√(2·cost)`)
- [x] **wasserstein/sliced** — `SlicedConfig {n_proj, p, seed}`, Box-Muller unit directions, equal-weight 1D `W_p^p` averaging
- [x] **wasserstein/max_sliced** — `MaxSlicedConfig`, argmax-init + finite-difference gradient ascent with re-projection to unit sphere
- [x] **gromov/gromov_wasserstein** — `GwConfig {eps, max_iter, inner_max_iter, tol}`, outer loop on gradient `G = −2·C₁·T·C₂^T` + inner Sinkhorn, Frobenius-norm convergence test
- [x] **gromov/fused** — `FgwConfig {alpha, gw}`, cost `M = (1−α)·C_xy + α·∇_GW(T)`
- [x] **unbalanced/unbalanced_ot** — `UnbalancedConfig {eps, tau_a, tau_b, max_iter, tol}`, generalised log-domain Sinkhorn with `f_i ← (τ_a/(τ_a+ε))·(ε log a_i − ε·LSE)`
- [x] **barycenter/free_support** — `BaryConfig`, alternating Sinkhorn + barycentric support update from λ-weighted-mean init
- [x] **barycenter/fixed_support** — `FixedBaryConfig`, Cuturi-Doucet `b ← Π_k (K_k^T·(a_k/(K_k·b)))^{λ_k}`
- [x] **jko/jko** — `JkoConfig {tau, eps, n_inner, tol}`, heat-equation prox step + closure-driven external-potential variant
- [x] **bridge/schrodinger** — `SchrodingerConfig`, log-domain IPF on `K = exp(−C/ε)`, marginal-violation convergence
- [x] **multi/multi_marginal** — `MmConfig`, log-domain tensor scaling alternating-axes update, `LSE_other(x)` excludes axis's own potential
- [x] **clustering/wasserstein_kmeans** — `WkmConfig`, W2-distance assignment + free-support barycenter centroid refinement
- [x] **domain/mapping** — `barycentric_map` (row-normalised plan applied to target supports), `ot_adapt` (Sinkhorn + map)
- [x] **metrics/metrics** — `marginal_violation`, `kl_divergence`, `js_divergence`, `transport_cost`, `entropy`
- [x] **PTX kernels** (ptx_kernels.rs) — 7 kernels × 6 SM versions: `sinkhorn_step_ptx`, `cost_matrix_ptx`, `transport_apply_ptx`, `sliced_proj_ptx`, `gromov_grad_ptx`, `unbalanced_step_ptx`, `barycenter_update_ptx`
- [x] **Integration tests** (e2e_tests.rs) — 18 e2e tests: Sinkhorn↔simplex agreement, marginals satisfied, large-ε uniform plan, Sinkhorn divergence self-zero, W1 translation invariance, W2 Dirac=Euclidean, sliced zero on equal samples, entropic-GW marginals, unbalanced τ→∞ matches balanced, Schrödinger-Bridge marginals, multi-marginal k=2 matches Sinkhorn, barycenter idempotence, kmeans run, barycentric-map mean-of-target, KL self-zero, handle constructs, PTX × 6 SM
- [x] **Benchmarks** (benches/ot_ops.rs) — PTX bench group (7 kernels × 4 SM = 28) + Sinkhorn 16×16 algo bench
- **Tests**: 155 passing

---

## Vol.45: Spiking Neural Networks [COMPLETE]

Pure-Rust spiking neural network library covering classical neuron models (LIF with `β·v + I` discrete-time + Hard/Soft reset, IF without leak, Izhikevich with two-half-step Euler + RS/FS/CH/IB presets, AdEx Brette-Gerstner with exp-clamp, stochastic Poisson rate), surrogate gradients (sigmoid `α·σ·(1−σ)` with two-branch stable formula, atan `α/(π(1+x²))`, triangle `max(0, 1−|x|/α)` with compact support, super-spike `α/(1+|x|·α)²`, fast-sigmoid), surrogate-gradient training (BPTT with surrogate dispatch, STBP with explicit reset gradient, SLAYER with truncated ε-PSP convolution kernel), plasticity (pair-STDP with exponential traces and weight clamping, R-STDP with eligibility traces gated by reward, triplet-STDP Pfister-Gerstner with long-window post-trace), ANN→SNN conversion (99-percentile threshold balancing, layer-chain propagation), input encoding (Bernoulli rate, TTFS latency, phase coding, Poisson rate), spiking layers (linear with Kaiming init, naive direct 2D conv, max/avg pool, recurrent with self-connections), Liquid State Machine (sparse random reservoir + spectral-radius rescaling via power iteration), and analytical metrics (firing rate, ISI, CV-ISI, van Rossum exp-filter L², Victor-Purpura DP, sync index).

### oxicuda-snn (~38 files, ~6,602 SLoC)
- [x] **neuron/lif** — `LifConfig {tau_m, v_th, v_rest, dt, reset:ResetMode}`, `LifState {v}`, `beta() = exp(−dt/τ_m)`, `lif_step` with Hard/Soft reset
- [x] **neuron/integrate_fire** — `IfConfig`, `IfState`, `if_step` (no leak, threshold + reset)
- [x] **neuron/izhikevich** — `IzhConfig` with `regular_spiking`/`fast_spiking`/`chattering`/`intrinsically_bursting` presets, two-half-step Euler `izh_step`, post-update clamp
- [x] **neuron/adex** — Brette-Gerstner defaults, `AdexConfig`/`AdexState`, `adex_step` with `(V−V_T)/Δ_T ≤ 50` exp-clamp
- [x] **neuron/poisson** — `poisson_step(rates, dt, rng, out)` with non-negative rate validation
- [x] **surrogate/sigmoid** — `stable_sigmoid` two-branch formulation + `sigmoid_grad`
- [x] **surrogate/atan** — `α/(π·(1+(α(v−v_th))²))`
- [x] **surrogate/triangle** — `max(0, 1−|v−v_th|/α)` with exact compact support
- [x] **surrogate/super_spike** — Zenke-Ganguli `α/(1+|v−v_th|·α)²`
- [x] **surrogate/fast_sigmoid** — `α/(1+|α(v−v_th)|)²`
- [x] **training/bptt** — `BpttConfig {t_steps, surrogate, alpha}`, `surrogate_eval` dispatcher, `bptt_unroll` with `dL/dv_t = surrogate'·dL/ds_t + β·dL/dv_{t+1}` and weight outer-product accumulation
- [x] **training/stbp** — explicit reset gradient `(1−s_t)·…` for hard reset; matches BPTT when no spikes occur
- [x] **training/slayer** — `SlayerConfig {tau_s, dt}`, `epsilon_psp` ε-kernel, truncated convolution `convolve_psp`, `slayer_loss` MSE
- [x] **plasticity/stdp** — `StdpConfig`, `StdpTraces`, pair-rule with exponential decay traces and `[w_min, w_max]` clamping
- [x] **plasticity/r_stdp** — `RStdpConfig`/`RStdpState`, eligibility-trace decay `e ← e·exp(−dt/τ_e) + STDP_event` gated by reward
- [x] **plasticity/triplet_stdp** — `TripletStdpConfig`/`TripletTraces`, additional long pre/post traces, reduces to pair STDP when `a2_*=0`
- [x] **conversion/ann2snn** — `SnnLayer`, `quantile`, `ann_to_snn_layer` 99-percentile rescale `W' = W·(λ_prev/λ); b' = b/λ`
- [x] **conversion/threshold_balance** — `balance_layer_chain` propagating `λ` across layer chain
- [x] **encoding/rate** — Bernoulli `out[t,i] = (rng < value[i])`, `rate_decode` time-average
- [x] **encoding/temporal** — TTFS `t_spike = floor((1−clamp(v, 0, 1))·(T−1))`
- [x] **encoding/phase** — phase-coded oscillatory reference signal
- [x] **encoding/poisson_input** — wraps `poisson_step` for input layers
- [x] **layer/spiking_linear** — `SpikingLinear` with Kaiming-normal init, `forward_step` (W·x + b → LIF)
- [x] **layer/spiking_conv** — `SpikingConv2d` naive direct sliding-window convolution + per-output-pixel LIF
- [x] **layer/spiking_pool** — `PoolKind {Max, Avg}`, `spike_pool` 2-D windowed reduction
- [x] **layer/recurrent** — `SpikingRecurrent` with `W_in·x + W_rec·s_{t-1}` + LIF, persistent `last_spikes`
- [x] **reservoir/lsm** — `LsmConfig {n_neurons, density, spectral_radius, w_in_scale, seed}`, `power_iteration_spectral_radius`, sparse-random `W_rec` rescaled to target ρ(W)
- [x] **metrics/metrics** — `firing_rate`, `isi`, `cv_isi`, `van_rossum_distance` (exp-filter L²), `victor_purpura_distance` (DP recurrence), `sync_index` (peak normalised cross-correlation)
- [x] **PTX kernels** (ptx_kernels.rs) — 7 kernels × 6 SM versions: `lif_step_ptx`, `surrogate_grad_ptx` (5-mode dispatch), `stdp_update_ptx`, `spike_conv_ptx`, `rate_encode_ptx`, `poisson_sample_ptx`, `bptt_accum_ptx`
- [x] **Integration tests** (e2e_tests.rs) — 8 e2e tests: LIF→sigmoid_grad chain, IF→atan_grad chain, Izh→super_spike chain, Poisson→LIF cascade, triangle/fast-sigmoid finite-grad, handle constructs, hard-vs-soft-reset divergence, PTX × 6 SM
- [x] **Benchmarks** (benches/snn_ops.rs) — PTX bench group (7 kernels × 4 SM = 28) + LIF-256 + sigmoid-grad-256 algo benches
- [x] **neuron/hodgkin_huxley** — `HhConfig` (squid-axon defaults), `HhState {v,m,h,n,spikes}`, `hh_step` (RK4 + exact exponential gating), `hh_run` (raster collection); `PrConfig` (CA3 defaults), `PrState`, `pr_step` Euler (somatic fast Na/K-DR + dendritic Ca²⁺/AHP, coupling gc/p)
- [x] **training/eprop** — `EpropConfig`, `EligibilityTraces {e, n_pre, n_post}`, `LearningSignal`; `update_eligibility_traces` (piecewise-linear pseudo-derivative + decay), `compute_weight_update` (task + firing-rate reg), `apply_weight_update` (optional gradient clip), `update_running_rates` (EMA), `decolle_learning_signals` (local readout error), `eprop_step` (full online update wrapper)
- **Tests**: 240 passing

---

## Future Work / Beyond v1.0

### Performance & Optimization
- [x] Stream-K GEMM scheduling (Hopper+ load balancing)
- [x] Persistent kernel GEMM (SM occupancy optimization)
- [x] Warp-specialization for Hopper TMA (Tensor Memory Accelerator)
- [x] FP6 / FP4 mixed-precision training kernels (precision/fp4_fp6_ops.rs) -- Blackwell sub-byte GEMM with micro-scaling
- [x] Graph-based kernel fusion (CUDA Graph equivalent)
- [x] Cooperative groups API

### Multi-GPU & Distributed
- [x] Multi-GPU support with peer-to-peer memory access
- [x] NCCL-equivalent collective operations (AllReduce, AllGather, ReduceScatter) -- comm.rs in oxicuda umbrella
- [x] NVLink/NVSwitch topology-aware communication (nvlink_topology.rs) -- topology discovery, optimal ring/tree, Dijkstra routing
- [x] Multi-node distributed training support (distributed.rs) -- TcpStore/FileStore rendezvous, gradient bucketing, ZeRO sharding
- [x] Pipeline parallelism primitives (pipeline_parallel.rs) -- GPipe, 1F1B, interleaved, zero-bubble schedulers
- [x] Distributed inference engine (oxicuda-dist-infer Vol.12) -- TP/SP/EP parallelism, distributed KV cache, prefix-affinity routing

### Additional Backends
- [x] AMD ROCm backend (HIP runtime) -- memory + compute ops (GEMM, Conv2D, Attention, unary, binary, reduce)
- [x] Intel oneAPI / Level Zero backend -- memory + compute ops (GEMM, Conv2D, Attention, unary, binary, reduce via OpenCL SPIR-V)
- [x] Apple Metal backend (via metal-rs) -- memory + compute ops (GEMM, Conv2D, Attention, unary, binary, reduce via MSL shaders)
- [x] WASM + WebGPU backend (oxionnx-web) -- memory + compute ops (GEMM, Conv2D, Attention, unary, binary, reduce via WGSL shaders)
- [x] Vulkan Compute backend -- memory + compute ops (GEMM, Conv2D, Attention, unary, binary, reduce via SPIR-V compute shaders)

---

## Backend Compute Operations [COMPLETE]

All 5 alternative GPU backend crates now have compute operations (GEMM, Conv2D, Attention, unary elementwise, binary elementwise, reduce) fully wired up instead of returning `Unsupported`.

### oxicuda-webgpu — WGSL Shader Dispatch via wgpu
- [x] GEMM compute shader (WGSL tiled matrix multiply)
- [x] Unary elementwise (relu, sigmoid, tanh, exp, log, sqrt, abs, neg — the full `UnaryOp` enum; no `gelu`/`silu` generator exists in this crate at all, unlike `oxicuda-metal`)
- [x] Binary elementwise (add, sub, mul, div, max, min — the full `BinaryOp` enum) with `binary_wgsl` generator — the generator also emits `pow`, but `BinaryOp` has no `Pow` variant so it is not reachable through the trait (see follow-ups)
- [x] Reduction (sum, max, min, mean) with `reduction_wgsl` + `reduction_final_wgsl`
- [x] Conv2D forward (WGSL NCHW compute shader + CPU fallback for shapes the fixed-literal 2D dispatch can't express)
- [x] Attention (WGSL scaled dot-product + stable softmax + causal masking, same CPU-fallback caveat)
- [x] Batched GEMM (`batched_gemm_wgsl` shader + trait override with 3D dispatch)
- [x] FP16 GEMM (`gemm_wgsl_f16` shader with `enable f16` + `gemm_f16()` method)
- 293 of 293 tests passing (measured 2026-08-12, `--all-features`, real Apple M3 hardware). Count moves quickly — re-run `cargo nextest run -p oxicuda-webgpu --all-features` for the current figure.

### oxicuda-metal — MSL Shader Dispatch via metal-rs
- [x] GEMM / batched GEMM compute shaders (MSL tiled matrix multiply) — **widened 2026-08-12**: dispatch now goes through a runtime-parameterised v2 kernel (`gemm_msl_v2`/`batched_gemm_msl_v2`) supporting all four transpose combinations; previously only `NoTrans` operands with packed leading dimensions were accepted
- [x] Unary elementwise (relu, sigmoid, tanh, exp, log, sqrt, abs, neg — the full `UnaryOp` enum) — MSL also generates `gelu`/`silu`, but `UnaryOp` has no such variants so they are not reachable through the trait (see follow-ups)
- [x] Binary elementwise (add, sub, mul, div, max, min — the full `BinaryOp` enum) with `binary_msl` generator — MSL also generates `pow`, unreachable for the same reason
- [x] Reduction (sum, max, min, mean) with dedicated MSL max/min/mean shaders, plus a two-pass chunked reduction (`chunked_reduce_msl`) for flat 1-D reductions of >= 4096 elements
- [x] Softmax (row-wise, numerically stable) — **fixed 2026-08-12**: this was a false completion — `ComputeBackend::softmax` inherited the trait's `Unsupported` default despite a complete stable-softmax MSL shader already shipping in the crate; it now dispatches `softmax_msl_with_mode` for real (last-axis only — an earlier axis returns `Unsupported` rather than a silently wrong reduction)
- [x] Conv2D forward — **fixed 2026-08-12**: this was also a false completion — the trait method was a pure-CPU scalar loop that round-tripped every operand through `copy_dtoh`/`copy_htod` even though a finished MSL kernel existed with no caller anywhere in the crate; it now dispatches `conv2d_msl_v2` unconditionally on the GPU (no host fallback)
- [x] Attention — **fixed 2026-08-12**: same prior CPU-loop false completion; now dispatches `attention_msl_v2` (single-pass online-softmax, FlashAttention-style) on the GPU, falling back to the host implementation only when `head_dim`'s accumulator cannot fit in one threadgroup-memory slice (a real, caller-reachable limit — the on-device test suite asserts the GPU path is actually taken)
- [x] Batched GEMM — see the GEMM line above (same v2 kernel family, full transpose support)
- [x] FP16 GEMM (`gemm_f16()` inherent method, dispatches the v2 kernel with `GemmDtype::F16`)
- 401 tests passing (measured 2026-08-12, `--all-features`, real Apple M3 hardware; concurrent work in this crate means this count moves quickly — re-run `cargo nextest run -p oxicuda-metal --all-features` for the current figure)

#### Honest follow-ups from the 2026-08-12 metal/webgpu audit

- [ ] **Real MPS dispatch** (`oxicuda-metal`) — `mps.rs` today is parameter-validation descriptors only, with no MPS framework call anywhere (confirmed still true this session). Wiring real dispatch needs a new macOS-only, default-off dependency on `objc2-metal-performance-shaders` (built on `objc2`/`objc2-metal`, Pure-Rust-policy-compliant system-framework interop), constructing real `MPSMatrixDescriptor`/`MPSMatrix`/`MPSMatrixMultiplication` objects and calling `encodeToCommandBuffer:leftMatrix:rightMatrix:resultMatrix:` against the `metal::CommandQueue` already owned by `MetalDevice`. Not started.
- [ ] **Async dispatch, default-on** (`oxicuda-metal`) — `MetalBackend::set_async_dispatch` now exists (command buffers commit-and-return, awaited at the next synchronisation point: `synchronize`, any host read/write, `free`, `launch_custom_kernel`, or backend drop) but defaults to **off** (synchronous, wait-per-op). Evaluate flipping the default once it has run longer in the wild — the correctness argument is already documented on the method, but a default flip is a behavior change worth its own soak period.
- [ ] **`UnaryOp::Gelu`/`Silu` and `BinaryOp::Pow`** (`oxicuda-backend`) — both `oxicuda-metal` (MSL) and `oxicuda-webgpu` (WGSL, `pow` only — it has no gelu/silu generator at all) already generate shader source for these ops, but the shared `UnaryOp`/`BinaryOp` enums have no variants for them, so none of the four alt-backends can reach them through `ComputeBackend`. Adding the variants is a breaking change to a trait ~70 crates depend on (every backend's `match` becomes non-exhaustive) — group it with the coordinated soundness/API batch above rather than doing it alone.
- [ ] **bf16 GEMM on Metal** — `oxicuda-metal::numeric` has host-side bf16 pack/unpack (round-to-nearest-even, tested), but there is no MSL `bfloat` GEMM kernel: it needs a Metal-3.1 device-family compile gate and `msl::GemmDtype` has no `Bf16` variant. Not started.
- [ ] **`simdgroup_gemm_msl_v2` is F32-only** — the F16 path returns `Err(Unsupported)` (`simdgroup_half8x8` would need to accumulate in `half`, which was judged not worth the added rounding-error surface without a concrete caller). Revisit if a caller needs it.
- [ ] **Managed-storage `did_modify_range` write path is compile-verified only** — `MTLStorageModeManaged` does not exist on Apple Silicon (this session's hardware, and every Mac since the M1), so the Managed-mode branch in `MetalMemoryManager` never executes in CI or in this audit. Needs an Intel-Mac (or virtualized-Intel) run to actually exercise it.
- [~] **`ComputeBackend::unary`'s in-place aliasing contract is undocumented, and the backends diverge** — `WebGpuBackend::unary` rejects `input_ptr == output_ptr` (`InvalidArgument`: wgpu forbids binding one buffer as both `read` and `read_write` in one dispatch), while `MetalBackend::unary` has no aliasing check at all and silently permits it. The trait doc says nothing either way. *Done:* both crates' `tests/gpu_presence.rs` now allocate separate input/output buffers with a comment explaining why, so neither presence test depends on the divergence — verified passing on real Apple M3 hardware (`--features gpu-tests`, 686/686). *Remaining:* document the aliasing contract on `ComputeBackend::unary` itself, and decide whether Metal should also reject it for cross-backend consistency. Note this is a trait-doc change only — the `ComputeBackend` signature is depended on by ~70 crates and must not change.
- [ ] **`oxicuda-webgpu` can abort the process under sustained unsynchronised submission** — `gemm`/`reduce`/`conv2d_forward`/`attention` never synchronise per operation by design, and issuing many of them with no interleaved `synchronize()` panics inside `wgpu-core` (`queue.rs`, "timed out waiting on last submission"). That is a whole-process abort, not a recoverable `Err`, so a caller cannot defend against it. No test covers this: nextest isolates each test in its own process, so per-process submission counts never get high enough to trigger it; it was reproduced from benchmark-shaped loops, and the in-tree Criterion benches work around it by synchronising every iteration. Needs either an internal submission-count watermark that forces a flush, or an explicit documented cap on outstanding submissions.
- [ ] **`crates/oxicuda/src/compute.rs` module doc is stale** — its "What gets selected" / macOS section still describes Metal as covering only `gemm`/`batched_gemm`/unary/binary/reduce, understating the `conv2d_forward`/`attention`/`softmax` fixes above. Needs a code-owner doc pass (out of scope for a docs-only file sweep — `compute.rs` is source, not one of `TODO.md`/`CHANGELOG.md`/a README).

### oxicuda-vulkan — SPIR-V Compute Shader Dispatch via ash
- [x] GEMM compute shader (SPIR-V tiled matrix multiply)
- [x] Unary elementwise (SPIR-V generator for all standard ops)
- [x] Binary elementwise (SPIR-V generator for all standard ops)
- [x] Reduction (SPIR-V generator for sum, max, min, mean)
- [x] Conv2D forward (SPIR-V NCHW compute shader + CPU fallback)
- [x] Attention (SPIR-V scaled dot-product + stable softmax + causal masking)
- [x] Batched GEMM (`batched_gemm_compute_shader` SPIR-V + trait override with 3D dispatch)
- 66 tests passing

### oxicuda-rocm — HIP Kernel String Generators + Host-Side Dispatch
- [x] GEMM kernel (HIP tiled matrix multiply)
- [x] Unary elementwise (HIP kernel generator)
- [x] Binary elementwise (HIP kernel generator)
- [x] Reduction (HIP kernel generator for sum, max, min, mean)
- [x] Attention kernel (HIP fused attention)
- [x] Conv2D kernel (HIP convolution)
- [x] Batched GEMM (`batched_gemm_hip` kernel + trait override with CPU fallback)
- 56 tests passing

### oxicuda-levelzero — OpenCL SPIR-V Kernel Dispatch
- [x] GEMM kernel (OpenCL SPIR-V generator)
- [x] Unary elementwise (OpenCL SPIR-V generator)
- [x] Binary elementwise (OpenCL SPIR-V generator)
- [x] Reduction (OpenCL SPIR-V generator for sum, max, min, mean)
- [x] Conv2D forward (OpenCL SPIR-V NCHW compute shader + CPU fallback)
- [x] Attention (OpenCL SPIR-V scaled dot-product + stable softmax + causal masking)
- [x] Batched GEMM (`batched_gemm_compute_shader` OpenCL SPIR-V + trait override with 3D dispatch)
- 69 tests passing

---

### Deep Learning Extensions
- [x] FlashAttention-3 with Hopper-specific optimizations
- [x] Speculative decoding attention kernels
- [x] RNN / LSTM / GRU cells
- [x] Deformable convolution
- [x] Transposed convolution (conv/transpose_conv.rs)
- [x] Sparse attention patterns (sliding window, dilated)
- [x] INT4 / NF4 quantization (QLoRA support)
- [x] Dynamic batching / continuous batching (dynamic_batch.rs) -- ContinuousBatcher, paged KV, speculative decoding

### Scientific Computing Extensions
- [x] Multi-GPU FFT (multi_gpu.rs) -- 1D slab decomposition across multiple devices
- [x] Sparse eigenvalue solver (Lanczos, Arnoldi)
- [x] Tensor decomposition (tensor_decomp.rs) -- CP-ALS, Tucker HOSVD/HOOI, TT-SVD
- [x] ODE/PDE solver kernels (ode_pde.rs) -- Euler, RK4, RK45, implicit Euler, BDF2, heat/wave/Poisson/advection
- [x] Monte Carlo simulation primitives (monte_carlo.rs) -- MC integration, MCMC (MH, HMC), financial MC, variance reduction

### Ecosystem Integration
- [x] ComputeBackend trait for SciRS2
- [x] OxiCudaComputeBackend implementation
- [x] OxiONNX GPU inference backend (onnx_backend/) -- IR graph, op implementations, executor, planner, fusion, shape inference
- [x] ToRSh GPU backend (tensor_backend/) -- tensor, dtype, autograd, ops, optimizer, mixed precision
- [x] TrustformeRS Transformer GPU backend (transformer_backend/) -- KV-cache, attention, scheduler, speculative decoding, sampling, quantization
- [x] Benchmarks suite (criterion) with CI regression tracking -- oxicuda/benches/ (ptx_generation, autotune_search, fft_planning, blas_dispatch, backend_operations)
- [~] Published documentation on docs.rs (2026-05-01)
  - **Status:** `[package.metadata.docs.rs]` added to all 34 subcrate Cargo.toml files; docs build cleanly with `cargo doc --no-deps --all-features` (zero errors, zero warnings)
  - **Remaining:** Actual publication requires `cargo publish` — pending `/bump` flow

### Tooling
- [x] oxicuda-prof -- GPU profiling and tracing tool (profiling hooks implemented in oxicuda/profiling.rs)
- [x] oxicuda-debug kernel debugging (debug.rs) -- KernelDebugger, MemoryChecker, NanInfChecker, PTX instrumenter
- [x] Visual PTX explorer (tui_explorer.rs) -- CFG visualization, register lifetimes, instruction mix, PTX diff, complexity metrics
- [x] Automatic kernel fusion pass in PTX IR

---

## Aggregated Blueprint Quality Gates

Summary of quality gates from all 5 blueprint volumes. Each crate's TODO.md contains full details.

**Completed gates (2026-04-11):** A1, A3, A4 (autotune); P4, P7 (ptx); S15 (rand); plus NIST SP 800-22 statistical tests, FlashAttention-3 Hopper kernel bodies, multi-GPU FFT executor, matrix function PTX kernel bodies, backward error formula tests, sparse auto-selection heuristics.

**Completed gates (2026-04-11, Batch 15):** A1, A2, A3, A4 (autotune); P4, P7, wgmma, mma, WMMA, ldmatrix, cp.async, 3/4-stage pipeline, FP4 E2M1, tcgen05, TMA/cp.async.bulk, cluster barrier, griddepcontrol (ptx); D1-D4 docs; driver cluster launch, TMA descriptors, sm_100/120 occupancy, error injection; launch tracing; DNN FlashAttn tile, Winograd 4×, GQA, FP8 quant, accuracy suite.

### Vol.1 — Foundation (oxicuda-driver, oxicuda-memory, oxicuda-launch)

| Gate | Requirement | Target | Status |
|------|-------------|--------|--------|
| F1–F5 | Dynamic load, multi-GPU, context, stream, E2E kernel | P0 | [ ] |
| F6–F8 | DeviceBuffer, PinnedBuffer, MemoryPool | P0 | [ ] |
| F9–F10 | Error handling, Drop resource release | P0 | [ ] |
| NF2 | H2D / D2H bandwidth | ≥ 95% of PCIe theoretical | [ ] |
| NF3 | Kernel launch overhead | < 1 μs above raw `cuLaunchKernel` | [ ] |
| NF4 | Memory leak detection | Zero leaks (`compute-sanitizer`) | [ ] |
| D1–D4 | Docs: README, architecture.md, API docs, examples | — | [x] |

### Vol.2 — PTX + Autotuner (oxicuda-ptx, oxicuda-autotune)

| Gate | Requirement | Target | Status |
|------|-------------|--------|--------|
| P1–P8 | PTX generation: vector_add, GEMM sm_80, GEMM sm_90, elementwise, reduction, Tensor Core MMA, cache, ptxas compat | P0–P1 | [ ] |
| A1–A5 | Autotuner: pruning, stability, DB round-trip, dispatcher, CLI | P0–P1 | [ ] |
| A6 | Best autotuned config performance | ≥ 80% cuBLAS GEMM on A100 | [ ] |

### Vol.3 — BLAS (oxicuda-blas)

| Gate | Requirement | Target | Status |
|------|-------------|--------|--------|
| G1–G9 | GEMM (F16/BF16/F32/F64/FP8), Batched GEMM, BLAS L1 | P0–P1 | [ ] |
| G10–G14 | BLAS L2/L3, elementwise, reduction, epilogue fusion | P0–P1 | [ ] |
| P1–P3 | GEMM F16/F32/F64 M=N=K=4096 | ≥ 95% cuBLAS | [ ] |
| P4 | Batched GEMM 1000×(256³) | ≥ 90% cuBLAS | [ ] |
| P5–P6 | Softmax 4096², axpy 10M elements | ≥ 90–95% cuDNN/cuBLAS | [ ] |

### Vol.4 — DNN (oxicuda-dnn)

| Gate | Requirement | Target | Status |
|------|-------------|--------|--------|
| D1–D10 | Conv2D, FlashAttention, PagedAttention, RoPE, MoE routing + fused | P0–P1 | [ ] |
| D11–D20 | LayerNorm, RMSNorm, BatchNorm, GroupNorm, Pooling, Quantize, fused ops | P0–P2 | [ ] |
| P1 | Conv2D ResNet-50 layer3 | ≥ 90% cuDNN | [ ] |
| P2 | FlashAttention seq=2048, d=128, FP16 | ≥ 90% FlashAttention-2 | [ ] |
| P3 | PagedAttention decode batch=32, seq=4096 | ≥ 85% vLLM | [ ] |
| P4–P8 | MoE, LayerNorm, RMSNorm, BatchNorm, fused Conv+BN+ReLU | ≥ 90–95% / 2× speedup | [ ] |

### Vol.5 — Scientific Computing (oxicuda-fft/sparse/solver/rand + oxicuda umbrella)

| Gate | Requirement | Target | Status |
|------|-------------|--------|--------|
| S1–S6 | FFT 1D/2D/3D/batched/arbitrary-size | P0–P1 | [ ] |
| S7–S8 | SpMV / SpMM CSR correctness | P0 | [ ] |
| S9–S14 | LU, QR, Cholesky, SVD, Eigenvalue, Batched SVD | P0–P1 | [ ] |
| S15 | Philox uniform + normal correctness | P0 | [ ] |
| S16–S21 | SciRS2, oxionnx, ToRSh, TrustformeRS integration + CI/CD | P0–P1 | [ ] |
| P1–P2 | FFT 1D (2²⁰) and 2D (1024²) | ≥ 90% and ≥ 85% cuFFT | [ ] |
| P3 | SpMV CSR | ≥ 85% cuSPARSE | [ ] |
| P4–P6 | LU, SVD, Cholesky | ≥ 85–90% cuSOLVER | [ ] |
| P7 | Philox 100M samples | ≥ 95% cuRAND | [ ] |
| P8 | SciRS2 E2E typical workflow | ≥ 5× vs CPU-only SciRS2 | [ ] |

---

## Cross-Crate Numerical Accuracy Summary

| Library | Precision | Tolerance |
|---------|-----------|-----------|
| oxicuda-blas (GEMM) | FP64 | < 1e-14 |
| oxicuda-blas (GEMM) | FP32 | < 1e-5 |
| oxicuda-blas (GEMM) | FP16 | < 1e-2 |
| oxicuda-blas (GEMM) | BF16 | < 5e-2 |
| oxicuda-blas (GEMM) | FP8 | < 1e-1 |
| oxicuda-dnn (Conv2D) | FP16 / FP32 | < 1e-2 / < 1e-5 |
| oxicuda-dnn (FlashAttn) | FP16 / FP32 | < 2e-2 / < 1e-5 |
| oxicuda-dnn (Norms) | FP16 / FP32 | < 1e-3 / < 1e-6 |
| oxicuda-fft (round-trip) | FP32 | < N × 1.19e-7 |
| oxicuda-fft (round-trip) | FP64 | < N × 2.22e-16 |
| oxicuda-sparse (SpMV) | FP32 / FP64 | < 1e-5 / < 1e-14 |
| oxicuda-solver (residual) | FP32 / FP64 | < 1e-5 / < 1e-12 |

---

## v1.0 Completion Criteria (Vol.5 Sec 10.3)

| # | Condition | Status |
|---|-----------|--------|
| 1 | All SciRS2 CUDA dependencies eliminated (pure OxiCUDA backend) | [ ] Verify |
| 2 | oxionnx GPU inference operational on OxiCUDA backend | [ ] Verify |
| 3 | Major benchmarks achieve ≥ 95% of cuBLAS / cuDNN / cuFFT / cuSOLVER | [ ] Verify |
| 4 | Zero external dependencies beyond NVIDIA GPU driver (Pure Rust) | [ ] Verify |
| 5 | CI/CD pipeline with GPU tests + performance regression detection (5% threshold) | [ ] Verify |
| 6 | Documentation and examples cover all public API | [ ] Verify |

---

## Post-v1.0 Roadmap (from Blueprints)

| Version | Theme | Key Features | Status |
|---------|-------|-------------|--------|
| v1.1 | Multi-GPU | NCCL equivalent, NVLink topology, pipeline parallelism, distributed training | ✓ Done |
| v1.2 | Training | Gradient checkpointing, mixed-precision training, optimizer states on GPU | ✓ Done |
| v1.3 | Blackwell | FP4 / FP6 compute, 5th-gen Tensor Core, sm_100 / sm_120 optimized paths | ✓ Done |
| v2.0 | AMD ROCm | HIP backend, same API surface, ROCm 5.x+ | ✓ Done |
| v2.1 | Intel oneAPI | SYCL backend, Intel GPU (Arc, Ponte Vecchio) | ✓ Done |
| v3.0 | WASM + WebGPU | Browser GPU compute via WebGPU API | ✓ Done |
| v3.1 | State Space Models | S4 / Mamba / Mamba-2 (SSD) / RWKV — linear-time attention alternatives | ✓ Done |

---

---

## v0.2.0 Roadmap — Algorithm Wave Targets

All items below are pre-vetted v0.2.0 deliverables. Each wave follows the established pattern:
2 target crates × 3 deep algorithms each, full unit+e2e tests, zero clippy warnings, no `unwrap()`.

### Wave AAA+56 [COMPLETE]

**oxicuda-ann** (Approximate Nearest Neighbour & Vector Search):
- [x] `ann/binary_quant.rs` — Binary PQ (1-bit per sub-vector): threshold at centroid means, Hamming distance via popcount, asymmetric score via soft-decode; 2× memory vs SQ8
- [x] `lsh/pqfastscan.rs` — PQFastScan (André et al. 2015): SIMD-friendly 4-bit LUT scan; pack 32 PQ codes per 128-bit register; 16-lane vectorised distance accumulation
- [x] `ivf/ivfadc.rs` — IVFADC residual coding: per-inverted-list rotation + residual PQ encoding; refine with ADC lookup during search; reduces quantisation error by 2–4 dB

**oxicuda-tabular** (Tabular Deep Learning):
- [x] `transformer/node.rs` — NODE (Neural Oblivious Decision Ensembles, Popov 2019): differentiable oblivious trees with entmax-split threshold learning + ensemble averaging; trainable via SGD
- [x] `diffusion/tabddpm.rs` — TabDDPM (Kotelnikov 2023): Gaussian DDPM for continuous features + multinomial diffusion for categoricals; denoising UNet with mixed-type data
- [x] `gan/ctgan.rs` — CTGAN (Xu et al. 2019): conditional GAN with mode-specific normalisation for imbalanced categoricals; training-by-sampling with PacGAN discriminator

### Wave AAA+57 [COMPLETE]

**oxicuda-bayes** (Bayesian Deep Learning):
- [x] `gp/deep_gp.rs` — Deep Gaussian Processes (Damianou-Lawrence 2013): doubly-stochastic VI with inducing-point GP layers; DSVI ELBO = Σ E_q[log p(y|f_L)] - Σ KL(q(uₗ)‖p(uₗ))
- [x] `gp/sparse_gp_fitc.rs` — Sparse GP FITC/PITC (Titsias 2009): inducing-points variational lower bound; ELBO = log 𝒩(y; KₙₘK_mm⁻¹μ, σ²I+Qₙₙ−Kₙₙ_diag) + sparse posterior update O(nm²)
- [x] `calibration/ece_classwise.rs` — Class-wise ECE / per-class reliability diagrams (Kull 2019): per-class calibration curve, static + adaptive binning, multiclass Brier decomposition

**oxicuda-cs** (Compressed Sensing & Sparse Recovery):
- [x] `greedy/lista.rs` — LISTA (Learned ISTA, Gregor-LeCun 2010): unrolled T-layer ISTA with shared W,S weights trained by supervision on (y,x*) pairs; inference O(Tn) vs O(kn/ε) classic ISTA
- [x] `robust_pca/rpcagd.rs` — RPCA-GD (Yi 2016 non-convex GD): L=UVᵀ factored form, projected GD on L+S=M under incoherence, O(r²n) per iter vs O(n²) PCP nuclear norm
- [x] `cs/smc_cs.rs` — Sequential compressed sensing (Ji-Xue-Carin 2008): Bayesian SMC with particle updates for streaming measurement recovery; particle likelihood weighting + resampling

### Wave AAA+58 [COMPLETE]

**oxicuda-anomaly** (Anomaly Detection):
- [x] `isolation/inne.rs` — INNE (Isolation Nearest-Neighbour Ensemble, Bandaragoda 2018): isolation probability via t-ball ratio iz(x,kNN) = d(x,kNN_1)/max{d(x,kNN_k)}, ensemble average; O(ψ log n)
- [x] `svdd/deep_sad.rs` — DeepSAD (Ruff 2020 semi-supervised): hypersphere loss with labeled normal anchor pulls + anomaly anchor pushes; η-weighted loss ∈ {-1,+1} supervision signal
- [x] `distance/lof_online.rs` — Online LOF (Pokrajac 2007): incremental O(k²) updates to k-NN graph, LRD, and LOF scores on point insertion/deletion; no full refit on each new sample

**oxicuda-ot** (Optimal Transport):
- [x] `wasserstein/neural_ot.rs` — Neural OT map (Makkuva 2020, Korotin 2021): input-convex neural network parameterisation of W2 Kantorovich potentials; gradient of network = transport map T*
- [x] `bridge/flow_matching.rs` — Conditional Flow Matching (Lipman 2022, Liu 2023): ODE flow x_t=t·x₁+(1-t)·x₀, velocity v_t trained to CFM target u_t|x₀,x₁=x₁-x₀; simulation-free training
- [x] `domain/dro_wasserstein.rs` — Distributionally Robust Opt (Esfahani-Kuhn 2018): Wasserstein ball constraints ρ(P,P̂)≤ε; dual reformulation as regularised ERM + Lagrangian penalty

### Wave AAA+59 [COMPLETE]

**oxicuda-tda** (Topological Data Analysis):
- [x] `complex/alpha.rs` — Alpha complex filtration (Edelsbrunner-Mucke 1994): Delaunay triangulation dual, simplex radius = circumsphere radius clipped to Voronoi cell; much sparser than Rips
- [x] `homology/multi_parameter.rs` — Multi-parameter persistence (Lesnick-Wright 2015): 2-parameter filtrations; presentation matrices via minimal free resolution; Hilbert function + fibered barcodes
- [x] `mapper/stable.rs` — Stable Mapper (Carrière-Oudot 2018): statistical bootstrap confidence bands on Mapper graph topology; Fréchet mean of persistence diagrams under Wasserstein metric

**oxicuda-survival** (Survival Analysis):
- [x] `cox/landmark.rs` — Landmarking approach (Van Houwelingen 2007): dynamic landmark supermodels; predict conditional survival P(T>t*|T>s, Z(s)) from pooled landmark datasets
- [x] `aft/restricted_spline.rs` — Restricted cubic spline baseline hazard (Royston 2002): natural splines on log(-log(S(t))) with boundary knots; smooth hazard without piecewise assumption
- [x] `nonparametric/npsurv_bayes.rs` — Nonparametric Bayesian survival (Ferguson 1973, Hjort 1990): Dirichlet process prior on F; posterior draws via stick-breaking + KM-compatible hazard atoms

### Wave AAA+60 [COMPLETE]

**oxicuda-cvx** (Convex Optimisation):
- [x] `scs/scs_solver.rs` — SCS-style unified conic solver (O'Donoghue 2021): operator splitting for LP/QP/SOCP/SDP under one ADMM framework; K-cones: {non-negative, SOC, PSD, exponential, power}
- [x] `dcp/expr_tree.rs` — DCP expression tree (Disciplined Convex Programming): atom library (max/min/log/exp/norm/quad_form) with curvature propagation; reduce to conic form for SCS dispatch
- [x] `riemannian/riemannian_cvx.rs` — Riemannian convex optimisation (Zhang-Sra 2016): Riemannian gradient descent + retraction on Stiefel/Grassmann/SPD manifolds; geodesic Armijo line search

**oxicuda-sketch** (Streaming Data Sketches):
- [x] `membership/bloomier.rs` — Bloomier filter (Chazelle 2004): function-valued Bloom encoding f: S→Σ with O(n) cells; off-by-one hashing with Σ-alphabet (not just {0,1}); lookup returns f(x) or ⊥
- [x] `cardinality/dp_hll.rs` — Differentially private HLL (Flajolet-privé 2022): Laplace/Gaussian mechanism on sketch registers; sensitivity analysis for Stochastic Averaging; (ε,δ)-DP guarantee
- [x] `stream/graph_sketch.rs` — Spectral graph sketch (Kelner-Levin 2013): Õ(n poly-log) sketch supporting spectral sparsification; row-sampling with effective-resistance weights

### Wave AAA+61 [COMPLETE]

**oxicuda-manifold** (Manifold Learning):
- [x] `riemannian/hyperbolic.rs` — Hyperbolic Poincaré ball model (Nickel-Kiela 2017): Riemannian SGD with Möbius addition/parallel transport; logarithmic/exponential maps; distance d = 2 arctanh(‖-u⊕v‖)
- [x] `riemannian/wrapped_normal.rs` — Wrapped Normal distribution (Nagano 2019): distributions on hyperbolic space via exponential map of Euclidean Normal; RSVI with automatic differentiation
- [x] `embedding/clustermap.rs` — ClusterMap (Damrich 2022): attraction/repulsion force-directed layout unifying t-SNE, UMAP, ForceAtlas2 under a single kernel/loss family; annealed temperature

**oxicuda-gnn** (Graph Neural Networks):
- [x] `gnn/sign.rs` — SIGN (Rossi 2020): scalable inception GNN; precompute Aᵏx for k=0..K, concat and process with MLP; no message-passing at inference; O(Kmd) preprocessing
- [x] `gnn/grand.rs` — GRAND (Chamberlain 2021): graph neural diffusion via continuous-time PDE dX/dt = div(G(X)∇X); implicit-explicit Euler + attention-based diffusivity G(X)
- [x] `gnn/graphsage_minibatch.rs` — GraphSAGE mini-batch (Hamilton 2017): neighbourhood sampling (uniform/degree-weighted); inductive mean/max/LSTM aggregators; Frontier sampling for GNNs at billion-node scale

### Wave AAA+62 [COMPLETE]

**oxicuda-numeric** (Numerical Analysis Primitives):
- [x] `quadrature/gauss_patterson.rs` — Gauss-Patterson sparse-grid quadrature: 1D nested Gauss-Kronrod rules (1,3,7,15,31,63,127 pts); multi-dimensional Smolyak sparse grid with Clenshaw-Curtis nodes
- [x] `ode/sdirk.rs` — SDIRK solvers (Singly Diagonally Implicit RK): Alexander 1977 SDIRK3/SDIRK4, stage-value Newton iterations, error control via embedded pairs; stiff-safe with fixed Jacobian
- [x] `diff/automatic_diff.rs` — Dual-number forward-mode automatic differentiation: `Dual<f64>` with overloaded arithmetic; Jacobian-vector products; composition-safe through transcendentals

**oxicuda-pde** (Numerical PDE Solvers):
- [x] `fem/p2_triangle.rs` — P2 quadratic finite elements: 6-DOF triangle (3 vertices + 3 edge midpoints); element stiffness/mass matrices via 3-point Gaussian quadrature; L2/H1 superconvergence
- [x] `spectral/fourier_3d.rs` — 3D pseudo-spectral Navier-Stokes (Canuto 2006): FFT-based Poisson projector + de-aliased 3/2-rule; RK4 time integration; incompressibility constraint via pressure correction
- [x] `dg/br2_elliptic.rs` — BR2 interior penalty scheme (Bassi-Rebay 1997): auxiliary variable formulation for 2nd-order elliptic operators in DG; consistent, symmetric, positive-definite discrete operator

### v0.2.0 Quality Gate (applied after each wave)

```bash
cargo nextest run --workspace --all-features 2>&1 | tail -5
cargo clippy --workspace --all-features -- -D warnings 2>&1 | tail -5
```

Both gates must be green (0 failures, 0 warnings) before a wave is logged in the status line.

---

## Maintenance

- [x] preemptive-splitrs-near-cap (planned 2026-05-01)
  - **Goal:** Prevent three files from crossing the 2000-line refactoring cap; split now while seams are clean
  - **Design:** Priority order: (1) `crates/oxicuda-sparse/src/ops/batched.rs` (1950 LoC, 665 test lines) — extract test module; (2) `crates/oxicuda/src/tensor_backend/ops.rs` (1986 LoC, 316-line test block) — extract test module; (3) `crates/oxicuda-blas/src/precision/fp4_fp6_ops.rs` (1955 LoC) — split along FP6/FP4 banner seam; use `splitrs` CLI for each; run clippy -D warnings immediately after each split
  - **Files:** Driven by `splitrs` output; verify with `rslines 50` post-split
  - **Tests:** Existing tests must pass unchanged after each split; `cargo nextest run -p <crate> --all-features`
  - **Risk:** Low for test-module extractions (prefer `mod tests { use super::*; }` pattern); medium for fp4_fp6_ops.rs if internal helpers need `pub(crate)` widening

---

## Vol.46: Differential Privacy [COMPLETE]

### oxicuda-privacy (~26 files, ~5,000 SLoC, 97 tests)
- [x] `error.rs` — `PrivacyError` enum (13 variants: InvalidParameter, EmptyInput, DimensionMismatch, BudgetExhausted, NonPositiveSensitivity, NonPositiveEpsilon, InvalidDelta, IndexOutOfRange, ConvergenceFailed, EmptyMechanismList, SvtQueryLimitExceeded, TreeDepthExceeded) + `PrivacyResult<T>`
- [x] `handle.rs` — `SmVersion`, `LcgRng` (MMIX 64-bit LCG + Box-Muller), `PrivacyHandle` with `generate_gaussian_noise` / `generate_laplace_noise`
- [x] `ptx_kernels.rs` — 7 kernels × 6 SM versions (75/80/86/89/90/100): `exponential_sample`, `laplace_noise`, `gaussian_noise`, `clip_gradient`, `svt_threshold`, `prv_convolve`, `oue_encode`
- [x] `mechanism/exponential.rs` — McSherry-Talwar (2007): P(i) ∝ exp(ε·qᵢ/(2Δq)), cumulative-weight sampling
- [x] `mechanism/report_noisy_max.rs` — Lap(Δq/ε) noise per score, argmax
- [x] `mechanism/propose_release.rs` — Propose-Test-Release (PTR): c = ln(1/2δ)/ε, Lap test on local sensitivity, release with Lap(Δ/ε) noise or abstain
- [x] `selection/sparse_vector.rs` — Streaming SVT (AboveThreshold): ε₁=ε/2 threshold noise, ε₂=ε/2 query noise, k-true-response budget via `SvtState`
- [x] `selection/above_threshold.rs` — Batch above-threshold: return indices of queries exceeding noisy threshold
- [x] `accounting/fdp.rs` — f-DP / GDP (Dong-Roth-Su 2022): trade-off function T_μ(α)=Φ(Φ⁻¹(1-α)-μ), `gdp_compose(mus)=√Σμᵢ²`, `gdp_to_epsilon_delta` via Φ approximation (Abramowitz-Stegun 7.1.26), `gaussian_mechanism_mu(Δ,σ)=Δ/σ`
- [x] `accounting/zcdp.rs` — zCDP (Bun-Steinke 2016): ρ=Δ²/(2σ²), composition ρ_total=Σρᵢ, (ε,δ) conversion ε=ρ+2√(ρ·ln(1/δ)); tCDP with truncation ω
- [x] `accounting/prv.rs` — PRV accountant (Gopi et al. 2021): Gaussian PRV pmf on uniform grid [grid_lo,grid_hi], O(n²) discrete convolution, `prv_delta(ε)`, `prv_epsilon(δ)` via binary search
- [x] `composition/advanced.rs` — Basic k·ε₀ / k·δ₀; strong composition (Dwork-Rothblum-Vadhan 2010): ε₀√(2k·ln(1/δ'))+k·ε₀(eε₀-1); heterogeneous composition
- [x] `composition/amplification_subsampling.rs` — Poisson subsampling: ln(1+q(eε-1)); uniform without-replacement: exact Balle et al. bound
- [x] `composition/amplification_shuffling.rs` — Erlingsson et al. 2019 shuffling bound: ε≤log(1+(eε₀-1)/(eε₀+1)·8√(2ln(4/δ)/n))
- [x] `optimizer/dp_ftrl.rs` — DP-FTRL with binary tree aggregation (Kairouz et al. 2021): noise tree of depth max_depth, path-based accumulation per step, FTRL update with L2 reg
- [x] `optimizer/dp_adam.rs` — DP-Adam: per-sample L2 gradient clip, aggregate + Gaussian noise, Adam β₁/β₂/ε moment updates
- [x] `local/grr.rs` — GRR k-ary: P(output=v|input=v)=eε/(eε+k-1), unbiased frequency estimator
- [x] `local/oue.rs` — OUE (Wang et al. 2017): one-hot encoding, per-bit Bernoulli flip (p=1/(eε+1)), unbiased frequency estimator
- [x] `local/rappor.rs` — RAPPOR simplified: Bloom-filter hash to k positions, per-bit flip at rate 1/(eε/k+1), frequency decode
- [x] `sensitivity/local_sensitivity.rs` — LS_mean, LS_median, LS_sum; calibrated noise addition via Lap(LS/ε)
- [x] `sensitivity/smooth_sensitivity.rs` — β-smooth sensitivity for mean (= 1/n global) and median (order-statistics walk); noise at scale S^β/(ε-β); β < ε validation
- [x] `metrics/metrics.rs` — `PrivacyBudget` tracker (spend/remaining/fraction), `gaussian_mse`, `snr_db`, `gaussian_utility`, `subsampling_amplification_factor`
- [x] `e2e_tests.rs` — 18 cross-module tests: exponential distribution correctness, RNM valid index, PTR finite release, SVT k-limit, GDP composition √3, GDP→(ε,δ) positive, zCDP additive, zCDP ordering, PRV delta monotone, strong < basic composition, Poisson amplification reduces ε, GRR unbiased, OUE unbiased, DP-Adam evolves, budget exhaustion, PTX×6SM
- [x] `benches/privacy_ops.rs` — Criterion: 7 PTX kernels × 4 SM versions + exponential_256 algo bench

---

## Vol.47: Hyperdimensional Computing [COMPLETE]

### oxicuda-hdc (~21 files, ~4,500 SLoC, 61 tests)
- [x] `error.rs` — `HdcError` (16 variants: ZeroDimension, DimensionMismatch, EmptyInput, ClassNotFound, ItemNotFound, InvalidNgramOrder, InvalidBinaryValue, FeatureIndexOutOfRange, EmptyItemMemory, AssocDimensionMismatch, CapacityExceeded) + `HdcResult<T>`
- [x] `handle.rs` — `SmVersion`, `LcgRng` (MMIX LCG, bit-32 extraction for booleans to avoid period-2 low-bit defect), `HdcHandle` with `random_binary_hv` / `random_integer_hv` / `random_complex_hv`
- [x] `ptx_kernels.rs` — 7 kernels × 6 SM versions: `xor_bind`, `bundle_majority`, `cyclic_shift`, `cosine_sim`, `hamming_dist`, `complex_bind`, `hd_classify` (string concatenation to avoid Rust 2024 format! conflict with PTX `%r` registers)
- [x] `vector/binary.rs` — `Vec<i8>` {±1}: `random_binary`, `validate_binary`, `binary_dot`, `bipolar_count`, `threshold_binary` (tie-breaking via LcgRng)
- [x] `vector/integer.rs` — `Vec<i32>` MAP model: `random_integer` (rem_euclid(3)-1 for uniform {-1,0,+1}), `integer_bind` (element-wise mult), `integer_bundle`, `integer_to_binary`, `integer_norm`
- [x] `vector/complex.rs` — FHRR: `Vec<f32>` length 2D interleaved [re₀,im₀,...]: `random_complex` (uniform phases), `complex_bind` (element-wise complex mult), `complex_conjugate`, `complex_bundle` + normalize, `complex_cosine`, `complex_normalize`
- [x] `ops/binding.rs` — `binary_bind` (sign product a*b), `binary_unbind` (same), `integer_bind_op`, `circular_convolution` O(n²), `circular_correlation` (flipped-a conv)
- [x] `ops/bundling.rs` — `bundle_binary` (majority vote + LcgRng tie-break), `bundle_integer` (element-wise sum), `bundle_complex` (complex sum + normalize), `weighted_bundle_binary`
- [x] `ops/permutation.rs` — `cyclic_shift`/`cyclic_shift_i32`/`cyclic_shift_f32` (left rotate by k), `cyclic_shift_right`, `random_permute`, `random_permutation` (Fisher-Yates), `inverse_permute`
- [x] `memory/item_memory.rs` — `ItemMemory`: symbol→HV store, NN query by dot-product, `add_random`, `contains`, `len`
- [x] `memory/assoc_memory.rs` — `AssocMemory`: bind-and-superpose; `store(key,val)` accumulates, `finalize()` thresholds to i8, `retrieve(key)` unbinds, `capacity_estimate()` = 0.138·D
- [x] `classifier/hd_classifier.rs` — `HdClassifier`: per-class i32 accumulator → thresholded prototype, argmax-cosine classify, error-corrective `online_update`
- [x] `classifier/prototype.rs` — `Prototype`: incremental add/subtract on i32 accumulator, `build()`, `cosine(query)`
- [x] `encoding/record.rs` — `RecordEncoder`: n_features × n_values_per_feature random HVs; `encode(feature_values)` = bundle(bind(feat_hv, val_hv))
- [x] `encoding/ngram.rs` — `NgramEncoder`: vocab HVs + cyclic shift; `encode(tokens)` = bundle over n-gram windows with order-j shifts
- [x] `encoding/pattern.rs` — `PatternEncoder`: row/col HVs; `encode(pixels, threshold)` = bundle active pixel positions; `encode_multilevel` over threshold array
- [x] `distance/hamming.rs` — `hamming_frac` = (D - Σaᵢbᵢ)/(2D), `hamming_count`, `hamming_similarity_threshold` (n_sigma·/√D)
- [x] `distance/cosine.rs` — `cosine_binary` via dot/D, `cosine_integer`, `cosine_real`, `cosine_complex` via Re(a·conj(b))/D, `argmax_cosine_binary`
- [x] `distance/jaccard.rs` — `jaccard_binary` (set intersection/union), `minihash_similarity` (correlation estimate)
- [x] `metrics/metrics.rs` — `hopfield_capacity` = ⌊0.138·D⌋ (Amit et al.), `classification_accuracy`, `bundle_snr` = √D/√k, `required_dimension` (birthday-paradox bound), `average_pairwise_hamming`
- [x] `e2e_tests.rs` — 22 cross-module tests: dimension correct, bind self-inverse, bundle recovers majority, shift roundtrip, item-memory exact match, assoc retrieval, classifier 2-class, online update, record/ngram/pattern distinguishability, Hamming/cosine self properties, complex bind-conjugate roundtrip, capacity, required_dimension scaling, handle generates valid HVs, PTX×6SM
- [x] `benches/hdc_ops.rs` — Criterion: 7 PTX kernels × 4 SM + binary_bind/cosine/bundle_16x algo benches

---

## Vol.48: Evolutionary & Genetic Algorithms [COMPLETE]

### oxicuda-evol (~28 files, ~5,000 SLoC, 18 tests)
- [x] `error.rs` — `EvolError` enum (14 variants: InvalidParameter, EmptyPopulation, DimensionMismatch, PopulationTooSmall, ConvergenceFailed, ObjectiveCountMismatch, EmptyGenome, InvalidInnovation, SpeciesNotFound, SwarmEmpty, PheromoneDimensionMismatch, TourIncomplete, EigenFailed) + `EvolResult<T>`
- [x] `handle.rs` — `SmVersion`, `LcgRng` (MMIX 64-bit LCG, bit-32 bool extraction, Box-Muller normal, Fisher-Yates shuffle), `EvolHandle`
- [x] `ptx_kernels.rs` — 7 kernels × 6 SM versions: `fitness_eval`, `tournament_select`, `gaussian_mutate`, `nsga_crowding`, `pso_update`, `de_mutate`, `cmaes_sample` (all via string concatenation, no format! for PTX registers)
- [x] `genetic/individual.rs` — `Individual{genome, fitness}`, `evaluate<F>`
- [x] `genetic/population.rs` — `Population{individuals, n_dims}`, random init, `evaluate_all`, `sort_by_fitness`, `best`
- [x] `genetic/selection.rs` — `tournament_select(k)`, `roulette_select`, `rank_select`
- [x] `genetic/crossover.rs` — `one_point_crossover`, `two_point_crossover`, `uniform_crossover(p_swap)`, `sbx_crossover(eta)` (Simulated Binary Crossover)
- [x] `genetic/mutation.rs` — `gaussian_mutate(σ, p_mut)`, `polynomial_mutate(η_m)`, `swap_mutate`
- [x] `evolution/cmaes/linalg.rs` — Jacobi eigendecomposition for symmetric n×n matrix (classical sweep algorithm, Givens rotations, sort by eigenvalue)
- [x] `evolution/cmaes/cmaes.rs` — Full CMA-ES (Hansen 2016): auto-param (λ=4+⌊3lnn⌋, μ=⌊λ/2⌋), rank-μ covariance update, p_c/p_σ cumulation paths, step-size CSA, h_σ indicator, eigendecomp scheduling every ⌊1/(c₁+c_μ)/n/10⌋ generations
- [x] `evolution/de/de.rs` — Differential Evolution: DE/rand/1, DE/best/1, DE/rand-to-best/1, DE/rand/2, DE/current-to-best/2; jDE self-adaptive F∈[0.1,1] and CR∈[0,1] per individual
- [x] `multiobjective/nsga2.rs` — `fast_nondominated_sort` O(MN²), crowding distance, binary tournament on (rank,crowding), SBX crossover, polynomial mutation, full `nsga2_run<F>`
- [x] `multiobjective/moead.rs` — Weight vector generation (linspace for 2D, regular simplex sampling for nD), Tchebycheff scalarization, T-neighbourhood by Euclidean weight distance, `moead_run<F>`
- [x] `neuroevolution/neat.rs` — NEAT: NodeGene/ConnectionGene, InnovationTracker (global counter + HashMap<(from,to),innov>), compatibility distance (c₁E+c₂D)/N+c₃W̄, speciation, Kahn topological sort for forward pass, evaluate_genome, structural mutation (add-node via split, add-connection), crossover
- [x] `swarm/pso.rs` — PSO: linear inertia decay w=[0.9→0.4], c₁=c₂=2.0, velocity clamp to v_max=0.1·(ub-lb), personal/global best, position clamp to bounds
- [x] `swarm/aco.rs` — ACO Ant System for TSP: τ₀=1/(n·greedy_len), roulette path construction (τ^α·η^β), global evaporation (1-ρ), pheromone deposit Q/L_k
- [x] `metrics/metrics.rs` — `hypervolume_2d` (sweep algorithm, exact), `igd`, `generational_distance`, `spacing` (std of NN distances), `average_nn_distance`, `extract_pareto_front`
- [x] `e2e_tests.rs` — 18 cross-module tests: GA binary max-ones, GA sphere converges, tournament pressure, CMA-ES Sphere 5D, CMA-ES Rosenbrock 2D, DE Sphere 5D, jDE adaptive, NSGA-II non-domination, NSGA-II front coverage, MOEA/D weight diversity, NEAT XOR, NEAT speciation, PSO Sphere 5D, PSO bounds, ACO 5-city TSP, hypervolume unit point, IGD ordering, PTX×6SM
- [x] `benches/evol_ops.rs` — Criterion: 7 PTX kernels × 4 SM + CMA-ES Sphere 5D algo bench

---

## Vol.49: Topological Data Analysis [COMPLETE]

### oxicuda-tda (~20 files, ~4,500 SLoC, 49 tests)
- [x] `error.rs` — `TdaError` enum (13 variants: EmptyPointCloud, DimensionMismatch, InvalidSimplex, ClosureViolation, EmptyComplex, FiltrationNotSorted, ReductionFailed, NanFiltrationValue, InvalidCoverParameter, LandmarkSelectionFailed, MatchingFailed, ParameterOutOfRange, DimensionTooLarge) + `TdaResult<T>`
- [x] `handle.rs` — `SmVersion`, `LcgRng` (MMIX LCG, bit-32 bool), `TdaHandle`
- [x] `ptx_kernels.rs` — 7 kernels × 6 SM: `pairwise_dist`, `filtration_sort`, `boundary_reduce`, `diagram_match`, `witness_dist`, `betti_count`, `mapper_cluster` (string concatenation)
- [x] `complex/simplex.rs` — `Simplex{vertices: Vec<usize>}` (sorted): `dim()`, `faces()`, `boundary()` (returns (±1, face) pairs, ∂²=0), `contains_face`
- [x] `complex/complex.rs` — `SimplicialComplex`: `add_simplex`, `add_simplex_with_closure`, closure verification, dimension indexing
- [x] `complex/filtration.rs` — `FilteredSimplex{simplex, value}`, `Filtration`: `vietoris_rips(dist, n_pts, max_radius, max_dim)` (0-simplices at 0, edges/triangles/higher at max-pairwise-dist), `vietoris_rips_from_points`, `sublevel_set`
- [x] `distance/pairwise.rs` — `pairwise_euclidean_sq`, `pairwise_euclidean`, `pairwise_manhattan`, `knn_graph`, `points_to_distance_matrix`
- [x] `homology/boundary.rs` — `BoundaryMatrix` (sparse Z₂ columns: Vec<Vec<usize>>): `from_filtration`, `low()`, `add_cols()` (XOR), `is_zero()`
- [x] `homology/reduction.rs` — ELZ 2002 standard persistence reduction: column addition over Z₂, pivot tracking via HashMap<row, col>
- [x] `homology/persistent.rs` — `PersistencePair{dim, birth, death: Option<f64>}`, `extract_persistence_pairs` from reduced boundary matrix + filtration
- [x] `persistence/diagram.rs` — `PersistenceDiagram{pairs, dimension}`: finite/essential filtering, `from_pairs_by_dim`
- [x] `persistence/barcode.rs` — `Bar{birth, death, dim}`, `Barcode`: `from_diagram`, `lifetimes`, `count_significant`, `n_finite`
- [x] `persistence/distance.rs` — `bottleneck_distance` (binary search + bipartite matching, exact), `wasserstein_1` (Hungarian O(n³)), diagonal distance
- [x] `mapper/mapper.rs` — `MapperGraph{nodes, edges}`: overlapping interval cover, Union-Find single-linkage clustering per interval, edge = shared point between clusters; `betti_1` = edges − nodes + components, BFS `connected_components`
- [x] `witness/witness.rs` — `maxmin_landmarks` (greedy farthest-point sampling), `lazy_witness_complex` (parameter R, 0th-NN distance m_w offset)
- [x] `metrics/metrics.rs` — `betti_numbers`, `persistent_entropy` H=−Σ(lᵢ/L)·log(lᵢ/L), `persistence_landscape` L_k(t)=k-th largest tent function, `landscape_distance` (L2), `total_persistence`, `count_components`
- [x] `e2e_tests.rs` — 19 cross-module tests: ∂²=0, Vietoris-Rips at r=0/r=∞, filtration sorted, 3-point H₁=1, boundary matrix reduces, birth≤death, 4-point square topology, bottleneck self=0, Wasserstein triangle inequality, Mapper circle loop, maxmin landmarks separated, witness vertices, Betti from diagram, entropy non-negative, landscape positive, essential classes, PTX×6SM
- [x] `benches/tda_ops.rs` — Criterion: 7 PTX kernels × 4 SM + Vietoris-Rips 20-point 2D bench

---

## Vol.50: Tensor Networks [COMPLETE]

### oxicuda-tn (~39 files, ~6,020 SLoC, 77 tests)
- [x] `error.rs` — `TnError` enum (13 variants: ShapeMismatch, DimensionMismatch, NotConverged, InvalidBondDimension, EmptyInput, IndexOutOfBounds, LinearAlgebraFailure, NumericalInstability, InvalidRank, UnsupportedSmVersion, InvalidConfiguration, RankExceedsLimit, ContractionPathInvalid) + `TnResult<T>`
- [x] `handle.rs` — `SmVersion`, `LcgRng` (MMIX 64-bit LCG, bit-32 bool, Box-Muller normal, range/categorical), `TnHandle`
- [x] `ptx_kernels.rs` — 7 kernels × 6 SM: `tensor_contract`, `svd_jacobi_step`, `dmrg_local_apply`, `mpo_apply`, `trotter_step`, `hosvd_unfold`, `tt_round` (string concatenation, no format! for PTX register names)
- [x] `svd/svd_dense.rs` — One-sided Jacobi SVD with sign-correct Givens rotations; m×n general matrices; sort singular values descending
- [x] `mps/tensor.rs` — `MpsTensor{shape:[D_l,d,D_r], data:Vec<f64> row-major}` with bounds-checked get/set and shape validation
- [x] `mps/mps.rs` — `MPS{site_tensors}`: `from_product_state` (computational-basis), `random_mps` (random bond tensors), `norm`, `local_expectation`, `rescale`, bond-mismatch detection
- [x] `mps/canonical.rs` — `left_canonicalize` / `right_canonicalize` via SVD; mixed canonical at bond `i`; ensures M^T·M = I (left) or M·M^T = I (right) within tol 1e-9
- [x] `mps/truncation.rs` — `svd_truncate(s, chi_max, tol)`, `bond_truncate` integrated with MPS canonicalisation
- [x] `mpo/mpo.rs` — `MpoTensor{shape:[D_l,d_out,d_in,D_r]}`, `MPO` struct, identity / Heisenberg constructors
- [x] `mpo/contraction.rs` — `apply_mpo_to_mps` with explicit bond fusion + SVD truncation
- [x] `dmrg/lanczos.rs` — Lanczos with full reorthogonalisation (Gram-Schmidt against prior vectors) + dense symmetric tridiagonal Jacobi eigensolver
- [x] `dmrg/dmrg.rs` — Two-site DMRG ground-state search with explicit left/right environment tensors, Lanczos local solver, SVD bond split + χ_max truncation, full left-right sweeps
- [x] `tebd/trotter.rs` — Suzuki-Trotter decomposition factors: 1st-order [1], 2nd-order Strang [½,1,½], 4th-order Suzuki nested
- [x] `tebd/tebd.rs` — Time-Evolving Block Decimation: per-bond gate application, even/odd alternation, SVD truncate to χ_max
- [x] `peps/peps.rs` — 2D `PEPS{shape:[D_l,D_r,D_u,D_d,d]}`, random init, corner indexing
- [x] `peps/contraction.rs` — Boundary-MPS approximate contraction (column-by-column)
- [x] `tt/tt.rs` — `TtTensor{cores}` representation, conversion from full tensor, reconstruction
- [x] `tt/tt_svd.rs` — Oseledets 2011 TT-SVD: sequential reshape → SVD → truncate → next core
- [x] `tt/tt_cross.rs` — TT-cross approximation via maxvol pivot selection
- [x] `tucker/hosvd.rs` — HOSVD: mode-k unfolding + SVD per mode, full-rank reconstruction within 1e-9
- [x] `tucker/hooi.rs` — HOOI alternating optimisation refining HOSVD core
- [x] `cp/als.rs` — CP/PARAFAC via ALS: Khatri-Rao + Hadamard-Gram normal equations; rank-1 recovery within 1e-6 residual
- [x] `contraction/einsum.rs` — Tiny einsum supporting binary contractions by labels (rejects duplicate labels in same tensor)
- [x] `contraction/path.rs` — Greedy contraction-cost path optimiser (flops + memory heuristic)
- [x] `metrics/metrics.rs` — `bond_dimension`, `entanglement_entropy` S = -Σ s²·ln(s²), `schmidt_spectrum`, `fidelity`, `product_state_zero_entropy`
- [x] `e2e_tests.rs` — 18 cross-module tests: product-state norm=1, left/right canonical orthonormality, MPO·MPS norm preservation, DMRG identity MPO energy, TEBD identity-gate norm conservation, TT-SVD/HOSVD roundtrip, CP-ALS rank-1 convergence, einsum matches manual loop, singlet entropy = ln 2, Lanczos smallest eigenvalue 5×5, SVD random reconstruction, fidelity self=1, greedy path, HOSVD truncated error finite, PTX×6SM
- [x] `benches/tn_ops.rs` — Criterion: 7 PTX kernels × all SM + MPS / SVD algo benches

---

## Vol.51: Sequence Models & Structured Prediction [COMPLETE]

### oxicuda-seq (~42 files, ~6,068 SLoC, 66 tests)
- [x] `error.rs` — `SeqError` enum (13 variants: ShapeMismatch, DimensionMismatch, NotConverged, InvalidConfiguration, EmptyInput, IndexOutOfBounds, NumericalInstability, ProbabilityOutOfRange, InvalidParameter, UnsupportedSmVersion, InvalidObservation, LengthMismatch, GraphInvariantViolated) + `SeqResult<T>`
- [x] `handle.rs` — `SmVersion`, `LcgRng` (MMIX 64-bit LCG, bit-32 bool, Box-Muller normal, `sample_categorical`), `SeqHandle`
- [x] `ptx_kernels.rs` — 7 kernels × 6 SM: `forward_pass`, `viterbi_step`, `crf_features`, `beam_topk`, `edit_dist`, `kalman_predict`, `mrf_gibbs` (string concatenation, inline LCG for Gibbs sampling)
- [x] `hmm/hmm.rs` — `HmmDiscrete{n_states, n_obs, pi, A, B}`, `HmmGaussian{n_states, pi, A, means, covs}` (diagonal cov)
- [x] `hmm/forward_backward.rs` — Log-space `forward`/`backward`, `log_likelihood`, posteriors γ and ξ; matches exhaustive enumeration to 1e-9
- [x] `hmm/viterbi.rs` — Log-space DP with backpointers; recovers ground-truth path for deterministic chains
- [x] `hmm/baum_welch.rs` — Full EM training: E-step (γ, ξ), M-step (π via γ_0, A via Σξ/Σγ, B via class-conditioned γ); log-likelihood non-decreasing
- [x] `crf/linear_chain_crf.rs` — `LinearChainCrf{n_labels, n_features, transitions, emissions}` score-form model
- [x] `crf/crf_train.rs` — Score-space forward-backward for partition / expected features; full L-BFGS two-loop recursion with Armijo backtracking + L2 reg; gradient finite-difference checked to 1e-3
- [x] `crf/viterbi_decode.rs` — Log-space Viterbi specialised for CRF score form
- [x] `memm/memm.rs` — Maximum Entropy Markov Model (per-state softmax over features), greedy + beam decoding
- [x] `ssvm/ssvm.rs` — `StructuredSvm` with loss-augmented Viterbi + sub-gradient training
- [x] `ssvm/cutting_plane.rs` — Constraint-based optimiser scaffold
- [x] `beam/beam.rs` — Generic `BeamSearch` over a scoring callback with length-normalisation + diversity penalty; matches exhaustive top-1 on tiny vocab
- [x] `alignment/needleman_wunsch.rs` — Global alignment, linear gaps; classic "GATTACA"/"GCATGCU" example
- [x] `alignment/smith_waterman.rs` — Local alignment; embedded-substring detection
- [x] `alignment/gotoh.rs` — Affine-gap 3-state DP (M / X / Y) with gap-open + gap-extend
- [x] `alignment/hirschberg.rs` — O(min(m,n))-memory NW via divide-and-conquer; identical score to NW (validated in e2e)
- [x] `grid_crf/grid_crf.rs` — 2D 4-connected pairwise CRF (image labelling)
- [x] `grid_crf/mean_field.rs` — Damped mean-field variational inference with neighbour message accumulation
- [x] `kalman/kalman_filter.rs` — Linear Kalman filter: innovation, S = HPH^T+R, K = PH^T S⁻¹, posterior update
- [x] `kalman/rts_smoother.rs` — Rauch-Tung-Striebel backward smoother; variance ≤ filter variance (enforced by test)
- [x] `kalman/ekf.rs` — Extended Kalman filter accepting boxed-closure Jacobians for f, h
- [x] `kalman/kalman_em.rs` — Shumway-Stoffer EM for Q/R covariance refit
- [x] `kalman/linalg.rs` — Local helpers: matmul, Gauss-Jordan inverse, Cholesky, determinant (kept private)
- [x] `mrf/mrf.rs` — General `Mrf` on graph + `IsingModel`; potential functions
- [x] `mrf/gibbs.rs` — Gibbs sampler with simulated-annealing temperature schedule; recovers low-T magnetisation
- [x] `mrf/belief_prop.rs` — Log-space loopy BP (sum-product marginals + max-product MAP) over factor graph
- [x] `metrics/metrics.rs` — `token_accuracy`, `sequence_accuracy`, Levenshtein `edit_distance` (`"kitten"→"sitting"=3`), BLEU-n with add-1 smoothing + brevity penalty, `perplexity`, `log_loss`
- [x] `e2e_tests.rs` — 15 cross-module tests: HMM forward-backward vs enumeration, Viterbi deterministic, Baum-Welch monotone, CRF Viterbi sanity, CRF gradient finite-diff, NW/SW/Gotoh/Hirschberg correctness, edit distance kitten→sitting, Kalman 1σ recovery, RTS smoother variance reduction, Ising Gibbs magnetisation, Beam top-1 vs enumeration, BLEU-1 identical = 1.0, PTX×6SM
- [x] `benches/seq_ops.rs` — Criterion: 7 PTX kernels × all SM + Viterbi / alignment algo benches

---

## Vol.52: Numerical PDE Solvers [COMPLETE]

### oxicuda-pde (~50 files, ~5,926 SLoC, 135 tests)
- [x] `error.rs` — `PdeError` enum (ShapeMismatch, NotConverged, EmptyMesh, InvalidGrid, DimensionMismatch, NumericalInstability, UnsupportedSmVersion, InvalidParameter, CflViolation, BoundaryConditionMissing, SingularMatrix, IndexOutOfBounds, …) + `PdeResult<T>`
- [x] `handle.rs` — `SmVersion`, `LcgRng` (MMIX 64-bit LCG, bit-32 bool, Box-Muller normal), `PdeHandle`
- [x] `ptx_kernels.rs` — 7 kernels × 6 SM versions: `fdm_stencil_5pt`, `gauss_seidel_step`, `csr_spmv`, `cg_axpy_dot`, `fem_assemble`, `mg_restrict`, `mg_prolong` (string concatenation only; no `format!` for PTX register lines)
- [x] `mesh/{mesh1d,mesh2d,triangulation}.rs` — Uniform 1D/2D grids; structured triangulation of a rectangle into 2 triangles per cell
- [x] `fdm/poisson_1d.rs` — −u''=f Dirichlet via Thomas tridiagonal; verified O(h²) convergence in e2e
- [x] `fdm/poisson_2d.rs` — 5-point Laplacian as CSR + Gauss-Seidel checkerboard; constant-RHS test
- [x] `fdm/heat_1d.rs` — Forward-Euler / Backward-Euler / Crank-Nicolson schemes; exponential-decay matches
- [x] `fdm/wave_1d.rs` — Leapfrog scheme with CFL stability check (returns `CflViolation` if violated)
- [x] `fdm/advection_1d.rs` — First-order upwind + second-order Lax-Wendroff with periodic BC mass conservation
- [x] `fem/p1_triangle.rs` — Linear Lagrange P1 element; local stiffness `K_e = (1/(4A))·B^T·B`, mass `M_e = (A/12)·[[2,1,1],[1,2,1],[1,1,2]]`
- [x] `fem/mass_stiffness.rs` — Global CSR assembly via per-triangle scatter, sparse pattern building
- [x] `fem/dirichlet_apply.rs` — Row/column zero + diagonal 1 Dirichlet enforcement
- [x] `spectral/chebyshev.rs` — Trefethen D₁ collocation matrix at `x_j = cos(jπ/N)`; exact differentiation of polynomial test
- [x] `spectral/fft_spectral.rs` — Real-to-complex DFT pseudo-spectral Poisson for periodic BCs; spectral accuracy on sin/cos
- [x] `time/{forward_euler,backward_euler,crank_nicolson,rk4,bdf2,imex}.rs` — Linear-decay validation; RK4 harmonic-oscillator energy conservation; IMEX splits implicit + explicit operators
- [x] `time/symplectic.rs` — Velocity Verlet (2nd-order) and Forest-Ruth (4th-order) symplectic integrators; phase-space volume preservation; energy near-conservation over long horizons
- [x] `time/sdirk.rs` — SDIRK2 (Alexander 1977, L-stable) and SDIRK3 (Crouzeix 1975, A-stable) with fixed-point implicit stage solve; stable for stiff ODEs with dt where explicit methods diverge
- [x] `multigrid/{smoother,restrict_prolong,vcycle}.rs` — Damped-Jacobi smoother; full-weighting (¼,½,¼) restriction; linear prolongation; V-cycle to analytic solution
- [x] `bc/{dirichlet,neumann,robin}.rs` — Dirichlet via row/column elimination; Neumann ghost-point; Robin α·u + β·∂u/∂n = γ
- [x] `solver/{cg,pcg,jacobi,ssor,ilu0,sparse}.rs` — Hestenes-Stiefel CG; PCG with Jacobi / SSOR / ILU(0) preconditioners; CSR mat-vec + dot + norm2 helpers
- [x] `dg/dg1d.rs` — Nodal DG with Legendre-Gauss-Lobatto nodes (Newton iteration to find roots); diagonal mass matrix; Lax-Friedrichs / upwind numerical flux
- [x] `metrics/metrics.rs` — L2 norm, H1 seminorm, max-norm, convergence-order estimate from refined grids
- [x] `e2e_tests.rs` — 22 cross-module tests: FDM Poisson 1D O(h²) convergence, Crank-Nicolson exponential decay, multigrid V-cycle convergence to analytic, FEM P1 Poisson, Chebyshev exact polynomial, FFT periodic Poisson, RK4 energy conservation, PCG (ILU0/SSOR/Jacobi) residual, Lax-Wendroff mass conservation, DG1D LGL quadrature exactness, PTX strings non-empty × 6 SM versions, …
- [x] `benches/pde_ops.rs` — Criterion: 7 PTX kernels × 4 SM + FDM Poisson 1D, Chebyshev, FFT-Poisson, CG, multigrid algo benches

---

## Vol.53: Manifold Learning & Riemannian Geometry [COMPLETE]

### oxicuda-manifold (~47 files, ~6,735 SLoC, 99 tests)
- [x] `error.rs` — `ManifoldError` enum (ShapeMismatch, NotConverged, EmptyInput, DimensionMismatch, InvalidParameter, EigenFailure, NumericalInstability, UnsupportedSmVersion, KNeighborsTooLarge, SingularMatrix, IndexOutOfBounds, …) + `ManifoldResult<T>`
- [x] `handle.rs` — `SmVersion`, `LcgRng` (MMIX 64-bit LCG, bit-32 bool, Box-Muller normal), `ManifoldHandle`
- [x] `ptx_kernels.rs` — 7 kernels × 6 SM versions: `pairwise_dist_sq`, `knn_topk`, `tsne_grad`, `umap_step`, `pca_center`, `mds_double_center`, `random_proj` (string-concatenation only)
- [x] `linear/pca.rs` — Center → covariance Σ = X^T X/(n−1) → Jacobi eigh → sort descending → project
- [x] `linear/kernel_pca.rs` — Gaussian / Polynomial / Linear kernels → centered Gram → eigh
- [x] `linear/fast_ica.rs` — Whitening + fixed-point iteration (tanh / gauss G), symmetric polar orthogonalisation
- [x] `tsne/perplexity.rs` — Per-row binary search for σᵢ to match target perplexity
- [x] `tsne/tsne.rs` — Full t-SNE: P→Q gradient with early-exaggeration + momentum; converges, separates clusters
- [x] `tsne/barnes_hut.rs` — 2D quadtree approximate gradient for n ≳ 1000
- [x] `umap/knn_graph.rs` — kNN edges + smooth-kNN σ/ρ fit (binary search to log₂(k))
- [x] `umap/fuzzy_simplicial.rs` — Membership μ in (0,1]; symmetrise via μ ∪ ν = μ + ν − μν
- [x] `umap/embedding.rs` — a/b curve fit + SGD with negative sampling; cross-entropy on edges
- [x] `local/lle.rs` — Constrained-LS weights with Σwᵢⱼ = 1 over kNN; M = (I−W)ᵀ(I−W); d+1 smallest eigenvectors, drop first
- [x] `local/mlle.rs` — Modified LLE with multi-weight basis (Zhang-Wang 2007)
- [x] `local/isomap.rs` — kNN graph + Dijkstra all-pairs geodesic distance + classical MDS
- [x] `local/laplacian_eigenmaps.rs` — Gaussian-weight W → normalised L_sym → generalised eigh `L v = λ D v` → drop constant eigenvector
- [x] `diffusion/diffusion_map.rs` — Coifman-Lafon: kernel + α density normalisation → row-stochastic P → eigh → `Ψᵢ = λᵢᵗ ψᵢ`
- [x] `mds/classical_mds.rs` — Torgerson: B = −½ J D² J → eigh → U√Λ
- [x] `mds/smacof.rs` — Iterative majorisation via Guttman transform
- [x] `neighbor/{knn_brute,kd_tree,ball_tree}.rs` — Brute / median-split KD-tree / centroid+radius Ball-tree neighbour search
- [x] `linalg/{jacobi_eig,power_iter,lanczos,householder_qr}.rs` — Cyclic Jacobi eigh; deflated power iteration; Lanczos with reorthogonalisation; Householder QR; polar orthogonalisation
- [x] `riemannian/stiefel.rs` — St(n,p): QR retraction; tangent projection `X − Y·sym(YᵀX)`
- [x] `riemannian/grassmann.rs` — Gr(n,p): principal-angle SVD geodesics
- [x] `riemannian/spd.rs` — SPD affine-invariant: exp_P(X) = P^½ exp(P^{−½}XP^{−½}) P^½ via symmetric matrix square roots
- [x] `riemannian/hyperbolic_poincare.rs` — Poincaré ball: Möbius addition + d(u,v) = arcosh(1 + 2‖u−v‖²/((1−‖u‖²)(1−‖v‖²)))
- [x] `optim/{riemannian_sgd,retraction}.rs` — Riemannian SGD on Stiefel + SPD
- [x] `metrics/metrics.rs` — Trustworthiness, continuity, KL(P‖Q), neighbourhood-preservation
- [x] `e2e_tests.rs` — 24 cross-module tests: PCA explained-variance, kernel-PCA isolates classes, FastICA recovers components, t-SNE separates clusters, UMAP fuzzy roundtrip, LLE swiss-roll-substitute, Isomap geodesic, classical MDS preserves distance, SMACOF monotone-stress, KD-tree vs brute consistency, Jacobi eigh orthogonality, Stiefel retraction stays-on-manifold, SPD exp/log roundtrip, Poincaré triangle inequality, PTX × 6 SM, …
- [x] `benches/manifold_ops.rs` — Criterion: 7 PTX kernels × all SM + PCA / Jacobi-eigh / kNN algo benches

---

## Vol.54: Statistical Inference & Hypothesis Testing [COMPLETE]

### oxicuda-stats (~72 files, ~6,412 SLoC, 160 tests)
- [x] `error.rs` — `StatsError` enum (ShapeMismatch, NotConverged, EmptyInput, InvalidParameter, NumericalInstability, UnsupportedSmVersion, InsufficientSampleSize, DegreesOfFreedomZero, ProbabilityOutOfRange, SingularMatrix, IndexOutOfBounds, …) + `StatsResult<T>`
- [x] `handle.rs` — `SmVersion`, `LcgRng` (MMIX LCG, bit-32 bool, Box-Muller normal), `StatsHandle`
- [x] `ptx_kernels.rs` — 7 kernels × 6 SM versions: `mean_var`, `rank_assign`, `histogram_bin`, `bootstrap_resample`, `permute_labels`, `chi2_cell`, `lr_normal_eq` (string concatenation only)
- [x] `special/{erf,gammaln,betainc,lgamma_series,digamma}.rs` — Abramowitz-Stegun 7.1.26 `erf` (validated `erf(0)=0`, `erf(1)≈0.8427`, `erf(2)≈0.9953`), Lanczos `lgamma` (validated `lgamma(5)=ln(24)`, `lgamma(0.5)=ln(√π)`), regularised incomplete beta via continued fraction (NR 6.4), regularised lower incomplete gamma `gammp`, asymptotic digamma
- [x] `distributions/{normal,student_t,chi_squared,f_dist,beta,gamma,binomial,poisson,exponential}.rs` — PDF/CDF/PPF for all; Student-t cdf at `t=0, ν=10` → exactly 0.5; ppf via Newton on cdf; normal ppf via Beasley-Springer-Moro
- [x] `descriptive/{summary,robust,quantile}.rs` — mean/var/stddev/skew/kurt; robust statistics (median, MAD, IQR, trimmed mean); empirical quantile types 1-9
- [x] `parametric/{t_test,anova,manova,regression_inference}.rs` — one-sample, two-sample Student, Welch (Satterthwaite df), paired t; one-way ANOVA (`{1,2,3},{3,4,5},{5,6,7}` → F=12.0 matches scipy), two-way (row/col/interaction SS); MANOVA Wilks-λ + Pillai trace + Hotelling-Lawley; regression SE/t/F/R²/adj-R²/AIC/BIC
- [x] `nonparametric/{mann_whitney,wilcoxon,kruskal_wallis,friedman}.rs` — rank-based with tied-rank averaging and normal/χ² approximations; signed-rank exact for n<25
- [x] `goodness_of_fit/{ks,anderson_darling,shapiro_wilk,jarque_bera}.rs` — KS one + two-sample with asymptotic Kolmogorov distribution; Anderson-Darling A²; Shapiro-Wilk W (Royston coefficients); Jarque-Bera χ²(2)
- [x] `chi_squared/{chi2_independence,fisher_exact,mcnemar}.rs` — r×c independence test; hypergeometric Fisher exact (one+two-sided); McNemar with continuity correction
- [x] `multiple/{bonferroni,holm,bh_fdr,by_fdr,tukey_hsd}.rs` — α/m; step-down Holm; BH/BY FDR; Tukey HSD via studentized-range approximation
- [x] `resampling/{bootstrap,jackknife,permutation}.rs` — B bootstrap replicates with statistic callback; leave-one-out jackknife variance; permutation test for group label shuffling
- [x] `ci/{normal_ci,t_ci,bootstrap_ci,proportion_ci}.rs` — normal-z, t, bootstrap percentile + BCa, Wilson + Clopper-Pearson + Agresti-Coull for proportions
- [x] `regression/{linear,logistic,ridge_lr}.rs` — OLS via Cholesky on normal equations; logistic via IRLS; ridge with λ regularisation
- [x] `power/{t_power,anova_power,effect_size}.rs` — sample size from `(d, α, β)`; η², partial η², ω² for ANOVA
- [x] `correlation/{pearson,spearman,kendall_tau}.rs` — Pearson r with t-test (df=n-2); Spearman via rank Pearson; Kendall τ via concordant/discordant
- [x] `e2e_tests.rs` — 18 cross-module tests: erf/lgamma boundary values, Student-t self-symmetry, ANOVA F-stat = 12.0, KS-1 normal small-D, Mann-Whitney identical groups → U=n·m/2, bootstrap CI contains true mean, OLS+inference returns expected SE/t/R², logistic IRLS classifies separable data, Wilks MANOVA two-group, Friedman ranks, Kendall τ on monotone, Wilson CI bracket coverage, Tukey HSD ordering, PTX × 6 SM, …
- [x] `benches/stats_ops.rs` — Criterion: 7 PTX kernels × all SM + erf / t-test / KS / OLS / bootstrap algo benches

---

## Vol.55: Streaming Data Sketches [COMPLETE]

### oxicuda-sketch (~57 files, ~6,210 SLoC, 176 tests)
- [x] `error.rs` — `SketchError` enum (InvalidParameter, EmptyStream, ShapeMismatch, DimensionMismatch, UnsupportedSmVersion, CapacityExceeded, IndexOutOfBounds, NumericalInstability, HashTableFull, DimensionMustBePowerOfTwo, …) + `SketchResult<T>`
- [x] `handle.rs` — `SmVersion`, `LcgRng` (MMIX LCG, bit-32 bool, Box-Muller normal), `SketchHandle`
- [x] `ptx_kernels.rs` — 7 kernels × 6 SM versions: `cm_update`, `cm_query`, `hll_register`, `bloom_insert`, `minhash_sketch`, `tdigest_merge`, `reservoir_sample` (string concatenation only)
- [x] `hash/{murmur3,fnv64,xxh3_min,universal,twouniv,tabulation}.rs` — Murmur3-32, FNV-1a 64, simplified xxH3 64, 2-universal `((ax+b) mod p) mod m` with `p = 2^61−1` Mersenne, tabulation hashing (per-byte 256-entry tables XOR)
- [x] `cardinality/{hll,hll_plus,linear_counting}.rs` — Flajolet HyperLogLog (`m=2^p`, `α_m·m²/Σ 2^(-Mⱼ)`, bias correction); HLL++ (Heule 2013: 6-bit registers + sparse rep + small-range table); linear counting for `n < 2.5·m`. HLL passes ±5% accuracy on 10000 distinct at p=14
- [x] `frequency/{count_min,count_sketch,conservative_update}.rs` — Cormode-Muthukrishnan CM (`d×w` table + 2-universal hashes, min over rows; over-estimate guaranteed); Charikar Count Sketch (sign hashes + median); conservative-update CM (update only minimum row)
- [x] `membership/{bloom,counting_bloom,cuckoo}.rs` — Bloom filter (m-bit + k hashes, optimal `k = (m/n)·ln(2)`, FP-rate ≈ `(1-e^(-kn/m))^k`, never false negative); counting Bloom (4-bit slots, supports deletion); Cuckoo filter (Fan et al. 2014, fingerprint + cuckoo hashing)
- [x] `quantile/{kll,t_digest,gk_quantile,p_square}.rs` — t-Digest (Dunning 2019, `k(q,δ)=δ·arcsin(2q-1)/(2π)` scale, merge-and-resize); KLL (Karnin-Lang-Liberty 2016, hierarchical compactors); Greenwald-Khanna ε-quantile (2001); Jain-Chlamtac P² with 5 markers + parabolic prediction
- [x] `topk/{misra_gries,space_saving,frequent}.rs` — Misra-Gries (k slots, ε=1/k-heavy-hitters); Metwally Space-Saving (replace min counter slot); frequent items > n/(k+1)
- [x] `similarity/{minhash,simhash,weighted_minhash}.rs` — K independent hash MinHash (Jaccard estimate converges to true value); Charikar SimHash (±w hyperplane votes → bit signature; cosine ≈ 1-2·hamming/d); Ioffe 2010 weighted MinHash (consistent weighted sampling)
- [x] `lsh/{cosine_lsh,jaccard_lsh,lsh_index}.rs` — Cosine LSH (K random hyperplanes → K-bit sig with L bands); Jaccard LSH (r×b banded over MinHash); generic LSH bucket-and-probe insert/query
- [x] `sampling/{reservoir,weighted_reservoir,bernoulli,priority}.rs` — Vitter reservoir (uniform-sample test passes); Efraimidis-Spirakis weighted reservoir (`key = u^(1/w)`, keep top-k); Bernoulli inclusion sampling; Duffield priority sampling
- [x] `moment/{ams_l2,johnson_lindenstrauss,lp_norm}.rs` — Alon-Matias-Szegedy L2 (Rademacher sketch + median-of-means); JL projection (Gaussian + Rademacher); Lp norm via stable random projections (Cauchy for L1, Gaussian for L2)
- [x] `stream/{online_mean_var,exponential_decay,sliding_window}.rs` — Welford online with Chan merge formula; exponential-decay weighted aggregates; sliding-window counts
- [x] `metrics/metrics.rs` — relative error, MAE, accuracy, recall-at-k
- [x] `e2e_tests.rs` — 22 cross-module tests: HLL accuracy 10000 distinct ±5% p=14, Count-Min over-estimate guarantee, Bloom false-negative-free, MinHash Jaccard convergence, t-Digest quantile within ε, KLL median accurate, Misra-Gries returns all heavy hitters, Space-Saving correct, reservoir uniform sample, weighted reservoir top-k, AMS L2-norm estimate, JL distance preservation, cosine LSH recall, Jaccard LSH, Welford online matches batch, PTX × 6 SM, …
- [x] `benches/sketch_ops.rs` — Criterion: 7 PTX kernels × all SM + HLL / Count-Min / Bloom / MinHash / reservoir algo benches

---

## Vol.56: Survival Analysis [COMPLETE]

### oxicuda-survival (~66 files, ~7,525 SLoC, 184 tests)
- [x] `error.rs` — `SurvivalError` enum (14 variants: ShapeMismatch, NotConverged, EmptyDataset, NoEvents, InvalidParameter, NumericalInstability, UnsupportedSmVersion, NegativeTime, SingularMatrix, IndexOutOfBounds, DimensionMismatch, …) + `SurvivalResult<T>`
- [x] `handle.rs` — `SmVersion`, `LcgRng` (MMIX LCG, bit-32 bool, Box-Muller normal), `SurvivalHandle`
- [x] `ptx_kernels.rs` — 7 kernels × 6 SM versions: `km_step`, `cox_risk_sum`, `cox_score`, `cox_info`, `logrank_oe`, `brier_score`, `rmst_integrate` (string concatenation only)
- [x] `data/{observation,dataset,risk_set}.rs` — `Observation{time, event}`, `Dataset` with optional covariates + strata; sorted-time risk set builder
- [x] `nonparametric/{kaplan_meier,nelson_aalen,life_table,survival_function}.rs` — KM `S(t) = Π(1-dᵢ/nᵢ)` + Greenwood `Var(log S) = Σ dᵢ/(nᵢ(nᵢ-dᵢ))` + log-log pointwise CIs; Nelson-Aalen `H(t) = Σ dᵢ/nᵢ` with variance; discrete-interval life table
- [x] `test/{log_rank,stratified_log_rank,peto_peto,gehan_breslow}.rs` — K-sample log-rank with hypergeometric variance, χ²(K-1); stratified summing O−E + V; Peto weight w_t=S(t); Gehan weight w_t=n_t
- [x] `cox/{cox_ph,breslow_ties,efron_ties,newton_raphson,schoenfeld,baseline_hazard}.rs` — Partial likelihood with Breslow + Efron tie handling; Newton-Raphson with line search; Cholesky on Fisher information; Schoenfeld residuals `rᵢ = xᵢ − x̄_R`; Breslow baseline `Ĥ₀(t) = Σ dᵢ/Σ exp(βᵀxⱼ)`
- [x] `aft/{weibull,log_normal,log_logistic,exponential,generalized_gamma,fit_aft}.rs` — Right-censored log-likelihood `Σ_events log f + Σ_censored log S`; Exponential closed-form MLE; Weibull, log-normal, log-logistic via Newton; generalized gamma via numerical-gradient ascent
- [x] `time_varying/{time_varying_cox,counting_process}.rs` — Counting-process formulation with `(start, stop, event, x(t))` intervals; risk-set membership based on (start, stop)
- [x] `competing/{fine_gray,cumulative_incidence,cause_specific_hazard}.rs` — CIF `F_k(t) = Σ S(tᵢ⁻)·dₖᵢ/nᵢ`; cause-specific Cox; Fine-Gray sub-distribution hazard with IPCW weights `w(t) = G(t)/G(tᵢ)`
- [x] `rmst/{rmst_estimator,restricted_mean}.rs` — `RMST(τ) = ∫₀^τ S(t)dt` via rectangle integration; delta-method variance
- [x] `concordance/{harrell_c,uno_c}.rs` — Harrell over comparable pairs; Uno IPCW-weighted variant
- [x] `calibration/{brier_score,ipcw_brier,integrated_brier,time_dependent_auc}.rs` — Naive Brier; IPCW Brier; integrated Brier over τ; time-dependent AUC (cumulative-incidence vs survivor)
- [x] `deep/{deepsurv_head,partial_likelihood_grad,surv_loss}.rs` — Gradient of Cox partial likelihood wrt log-risk η for DL head; `cox_loss`, `brier_loss` PyTorch-style callables
- [x] `linalg/{matmul,cholesky,solve,inverse}.rs` — Private Cholesky, Gauss-Jordan inverse, matmul, determinant
- [x] `special/{gammaln,digamma}.rs` — Lanczos `gammaln`, asymptotic `digamma`, Acklam normal-inverse
- [x] `metrics/metrics.rs` — Median survival, restricted mean, S(t) at horizon
- [x] `e2e_tests.rs` — 30 cross-module tests: KM exact recovery, Greenwood SE, Cox β recovery within 5%, Newton-Raphson <50 iter, Schoenfeld sum = 0, log-rank permutation invariance, Harrell C = 1.0 perfectly ranked / ≈ 0.5 random, RMST on constant S, Fine-Gray reduces to Cox without competing events, Weibull MLE on exponential recovers k≈1, PTX × 6 SM
- [x] `benches/survival_ops.rs` — Criterion: 7 PTX kernels × all SM + KM / Cox-Newton / log-rank / RMST algo benches

---

## Vol.57: Convex Optimisation [COMPLETE]

### oxicuda-cvx (~64 files, ~6,549 SLoC, 139 tests)
- [x] `error.rs` — `CvxError` enum (14 variants: NotConverged, ShapeMismatch, Infeasible, Unbounded, InvalidParameter, NumericalInstability, UnsupportedSmVersion, SingularMatrix, IndexOutOfBounds, DimensionMismatch, EmptyInput, ConeViolation, …) + `CvxResult<T>`
- [x] `handle.rs` — `SmVersion`, `LcgRng` (MMIX LCG, bit-32 bool, Box-Muller normal), `CvxHandle`
- [x] `ptx_kernels.rs` — 7 kernels × 6 SM versions: `axpy`, `soft_threshold`, `simplex_proj`, `gradient_step`, `fista_extrapolate`, `admm_dual_update`, `proj_l2_ball` (string concatenation only)
- [x] `lp/{revised_simplex,primal_dual_lp,mehrotra}.rs` — Revised simplex with Bland's rule + LU-of-basis updates; Mehrotra predictor-corrector primal-dual IP with centring `σ = (μ_aff/μ)³` + step cap α=0.99
- [x] `qp/{active_set_qp,primal_dual_qp}.rs` — Active-set with Schur-complement KKT; primal-dual IP for QP
- [x] `socp/primal_dual_socp.rs` — Alternating projection (cone + affine) with dual ascent over `(t,x): ‖x‖₂ ≤ t`
- [x] `sdp/{sdp_interior_point,log_det_barrier}.rs` — Newton on `−log det X` with PSD projection
- [x] `admm/{admm,consensus_admm}.rs` — Vanilla x/z/u updates; consensus ADMM for separable f = Σf_i
- [x] `proximal/{prox_gradient,fista,accelerated,douglas_rachford}.rs` — Proximal gradient with backtracking; FISTA momentum `t_{k+1} = (1+√(1+4t_k²))/2`; Nesterov accelerated; Douglas-Rachford `y←prox_f, z←prox_g(2y-x), x←x+z-y`
- [x] `primal_dual/chambolle_pock.rs` — Primal-dual extrapolation for `min f(Kx) + g(x)` with `τσ‖K‖² < 1`
- [x] `prox_ops/{l1,l2,linf,group_lasso,elastic_net,nuclear,total_variation_1d,indicator}.rs` — Soft-threshold L1; Tikhonov L2; L∞ Moreau dual; group-block soft-threshold; nuclear via Jacobi SVD soft-threshold on singular values; Condat O(n) 1D-TV; indicator-of-set
- [x] `projection/{simplex,l1_ball,l2_ball,box_proj,psd_cone,soc_cone,halfspace}.rs` — Wang-CP O(n log n) simplex/L1-ball; L2-ball `x·min(1, r/‖x‖)`; box clamp; PSD via Jacobi eigh + clip neg eigvals; SOC `(t,x)`; halfspace `aᵀx ≤ b`
- [x] `augmented_lagrangian/alm.rs` — Method of multipliers for equality constraints
- [x] `gradient/{projected_gradient,accelerated_gd,momentum_gd}.rs` — Projected GD `x ← Π_C(x − α∇f)`; Nesterov; Polyak heavy-ball
- [x] `linesearch/{armijo,wolfe,strong_wolfe,backtracking}.rs` — Armijo `f(x + αd) ≤ f(x) + c₁α∇fᵀd`; Wolfe; strong Wolfe with `|∇f(x+αd)ᵀd| ≤ c₂|∇fᵀd|`
- [x] `linalg/{cg,matvec,cholesky,qr,solve}.rs` — Private dense CG, LU, Cholesky, Householder QR
- [x] `metrics/metrics.rs` — Duality gap, primal/dual residual, KKT residual, convergence-rate estimator
- [x] `e2e_tests.rs` — 39 cross-module tests: LP 2D recovers vertex with -1 objective; QP identity-constrained returns 1; L1 prox `[2, 0.5, -0.5, -2] → [1, 0, 0, -1]`; simplex projection of `[1,1,1]` = `[1/3,1/3,1/3]`; PSD projection of `[-1,0;0,1] → [0,0;0,1]`; TV-1D denoising reduces stair-step; FISTA L1-LS O(1/k²) rate; ADMM-Lasso matches FISTA; Chambolle-Pock TV-L2 monotone primal energy; projected GD on box quadratic → KKT; strong Wolfe satisfies both conditions; PTX × 6 SM
- [x] `benches/cvx_ops.rs` — Criterion: 7 PTX kernels × all SM + LP / FISTA / ADMM / Chambolle-Pock algo benches

---

## Vol.58: Compressed Sensing & Sparse Recovery [COMPLETE]

### oxicuda-cs (~63 files, ~7,240 SLoC, 108 tests)
- [x] `error.rs` — `CsError` enum (14 variants: NotConverged, ShapeMismatch, InvalidParameter, NumericalInstability, UnsupportedSmVersion, SingularMatrix, IndexOutOfBounds, DimensionMismatch, EmptyInput, SupportTooLarge, InvalidSparsity, InvalidRank, InvalidConfiguration, RecoveryFailed) + `CsResult<T>`
- [x] `handle.rs` — `SmVersion`, `LcgRng` (MMIX LCG, bit-32 bool, Box-Muller), `CsHandle`
- [x] `ptx_kernels.rs` — 7 kernels × 6 SM versions: `correlate`, `hard_threshold`, `soft_threshold`, `iht_step`, `amp_onsager`, `svt_threshold`, `tv_grad` (string concatenation only)
- [x] `greedy/{omp,stomp,romp,cosamp,sp}.rs` — OMP atom selection via max `|aⱼᵀr|` + LS on support; StOMP stagewise with `t·σ̂` threshold; ROMP regularised; Needell-Tropp CoSaMP merge-prune; Subspace Pursuit
- [x] `thresholding/{iht,niht,htp,aiht}.rs` — Blumensath-Davies IHT `x ← H_K(x + μΦᵀ(y - Φx))`; NIHT adaptive `μ = ‖g_S‖²/‖Φ_S g_S‖²`; HTP with exact LS on identified support; AIHT with Nesterov momentum
- [x] `amp/{amp,vamp,eb_amp}.rs` — Donoho-Maleki-Montanari AMP with Onsager correction `b·z_{t-1}/M`; Rangan-Schniter-Fletcher VAMP; empirical-Bayes AMP
- [x] `basis_pursuit/{basis_pursuit,dantzig_selector}.rs` — BP `min ‖x‖₁ s.t. Φx=y` (ADMM); BPDN noisy; Candès-Tao Dantzig Selector via LP
- [x] `lasso/{coord_descent,lars,fista_lasso,group_lasso,fused_lasso,elastic_net}.rs` — Friedman cyclic coord-descent with warm-start path; Efron-Hastie LARS piecewise-linear path; FISTA on L1-LS; Yuan-Lin group LASSO per-block soft-threshold; fused LASSO `λ₂Σ|xⱼ-xⱼ₋₁|`; elastic net
- [x] `tv/{tv_1d_chambolle,tv_2d_chambolle,total_variation_denoise}.rs` — Chambolle 2004 primal-dual on dual; 2D anisotropic + isotropic TV
- [x] `matrix_completion/{svt,nuclear_norm,admm_completion}.rs` — Cai-Candès-Shen SVT `Y_{k+1} = Y_k + δ·P_Ω(M - X_k)` + `X_{k+1} = D_τ(Y)`; nuclear-norm min `min ‖X‖_* s.t. P_Ω(X) = P_Ω(M)`; ADMM completion
- [x] `robust_pca/{robust_pca_pcp,godec}.rs` — Candès-Li-Ma-Wright PCP `min ‖L‖_* + λ‖S‖₁ s.t. L+S=M`; Zhou-Tao GoDec bilateral random projections
- [x] `sparse_pca/sparse_pca_witten.rs` — Witten-Tibshirani-Hastie 2009 penalised SVD with L1 on loadings
- [x] `sbl/{sparse_bayesian,fast_marginal_likelihood}.rs` — Tipping 2001 RVM-style with per-coefficient hyperparameter γᵢ; Tipping-Faul 2003 fast marginal likelihood
- [x] `dictionary/{k_svd,mod_dl,online_dl}.rs` — Aharon-Elad-Bruckstein K-SVD with sparse-coding (OMP) + per-atom SVD update; Engan MOD; Mairal online DL
- [x] `measurement/{gaussian_matrix,bernoulli_matrix,partial_fourier,rip_estimator}.rs` — `Φᵢⱼ~N(0,1/m)`; `Φᵢⱼ~±1/√m`; random DFT row selection; RIP constant via random K-subset SVD
- [x] `linalg/{jacobi_svd,qr_householder,cholesky,lsqr,normal_equations}.rs` — Private Jacobi SVD, Householder QR, Cholesky, LSQR (Paige-Saunders), normal equations
- [x] `metrics/metrics.rs` — sparsity, recovery_error, support_recovery_rate, MSE, NMSE, PSNR, SNR
- [x] `e2e_tests.rs` — 27 cross-module tests: OMP recovers K=3 from m=20 n=50 Gaussian; CoSaMP support recovery ≥80%; IHT converges; AMP matches LASSO on iid Gaussian; BP exact when K<m/2; LASSO coord-descent ≈ LARS; TV denoising improves MSE on piecewise-constant + noise; SVT recovers low-rank from random sampling; PCP recovers L+S; soft-threshold `[2,0.5,-0.5,-2]` λ=1 → `[1,0,0,-1]`; hard-threshold top-2 of `[3,1,4,1,5]`; PTX × 6 SM
- [x] `benches/cs_ops.rs` — Criterion: 7 PTX kernels × all SM + OMP / FISTA-LASSO / SVT / TV / Robust-PCA algo benches

---

## Vol.59: Classical Graph Algorithms [COMPLETE]

### oxicuda-graphalg (~78 files, ~7,178 SLoC, 139 tests)
- [x] `error.rs` — `GraphalgError` enum (InvalidGraph, NegativeWeight, NegativeCycle, SourceOutOfRange, DisconnectedGraph, NotABipartiteGraph, NotADag, InvalidParameter, NumericalInstability, UnsupportedSmVersion, IndexOutOfBounds, EmptyInput, NotImplemented, …) + `GraphalgResult<T>`
- [x] `handle.rs` — `SmVersion`, `LcgRng` (MMIX LCG, bit-32 bool, Box-Muller), `GraphalgHandle`
- [x] `ptx_kernels.rs` — 7 kernels × 6 SM versions: `bfs_level`, `dijkstra_relax`, `pagerank_step`, `fw_inner`, `triangle_count`, `csr_spmv_bool`, `community_label` (string concatenation only)
- [x] `repr/{adjacency_list,adjacency_matrix,edge_list,csr_graph,weighted_graph}.rs` — 5 representations; conversions; standard accessors
- [x] `traversal/{bfs,dfs,iddfs,bidirectional_bfs}.rs` — BFS queue + visited; DFS recursive + iterative (explicit stack) + pre/post order; IDDFS memory-bounded; bidirectional BFS meet-in-the-middle
- [x] `topological/{kahn,dfs_topo}.rs` — Kahn in-degree queue; DFS post-order reverse; cycle detection in both
- [x] `shortest_path/{dijkstra,bellman_ford,spfa,floyd_warshall,johnson,a_star,yen_k_shortest,bidijkstra}.rs` — Dijkstra binary heap (rejects negative weights); Bellman-Ford O(VE) negative-cycle detection; SPFA queue-based; Floyd-Warshall O(V³); Johnson reweighting via Bellman-Ford then Dijkstra-per-source; A* admissible heuristic; Yen k-shortest via deviation method; bidirectional Dijkstra
- [x] `mst/{prim,kruskal,boruvka,union_find}.rs` — Prim priority queue from source; Kruskal sorted edges + Union-Find with union-by-rank + path compression; Borůvka contraction
- [x] `max_flow/{edmonds_karp,dinic,push_relabel,min_cut}.rs` — Edmonds-Karp BFS-augmenting O(VE²); Dinic level-graph + blocking-flow DFS O(V²E); push-relabel; min-cut via reachability in residual
- [x] `matching/{hopcroft_karp,hungarian_munkres,blossom_v_simple}.rs` — Hopcroft-Karp bipartite O(E√V); Kuhn-Munkres Hungarian O(n³); simplified Blossom for small general graphs
- [x] `connectivity/{scc_tarjan,scc_kosaraju,scc_gabow,bridges_tarjan,articulation_points,biconnected}.rs` — Tarjan low-link; Kosaraju two-pass; Gabow two-stack; Tarjan bridges low-link; articulation points; biconnected components edge-stack
- [x] `centrality/{degree_centrality,betweenness_brandes,closeness,eigenvector,pagerank,katz}.rs` — Brandes O(VE) BFS+dependency backprop; PageRank power iter with damping; Katz `(I-αA)⁻¹β`; eigenvector via power iteration; closeness `1/Σd(v,u)`
- [x] `community/{louvain,label_propagation,girvan_newman}.rs` — Blondel Louvain two-phase modularity; label propagation iterative most-frequent-neighbour; Girvan-Newman edge-betweenness removal
- [x] `arborescence/chu_liu_edmonds.rs` — Minimum spanning arborescence for directed graphs
- [x] `isomorphism/vf2.rs` — State-space search with backtracking + feasibility checks
- [x] `coloring/{greedy_coloring,dsatur,welsh_powell}.rs` — Greedy by index; DSATUR degree-of-saturation; Welsh-Powell degree-descending
- [x] `tsp/{christofides_approx,nearest_neighbor,two_opt}.rs` — Christofides 1.5-approx (MST + odd-degree perfect matching + Eulerian); NN greedy; 2-opt local search
- [x] `eulerian/hierholzer.rs` — O(E) Eulerian circuit construction
- [x] `hamiltonian/held_karp_dp.rs` — O(n²·2ⁿ) exact TSP DP
- [x] `metrics/metrics.rs` — Diameter, radius, density, global clustering coefficient = 3·triangles/triplets, transitivity
- [x] `e2e_tests.rs` — 30 cross-module tests: BFS line-graph distances; DFS tree visits all; Dijkstra 4-node correct; Bellman-Ford detects negative cycle; Floyd-Warshall = Dijkstra-all-pairs on positive; Prim = Kruskal total weight; Edmonds-Karp = min-cut; Tarjan SCC `[0→1,1→2,2→0,3→1]` = `{0,1,2},{3}`; PageRank Σ=1; Louvain modularity≥0; A* zero-heuristic = Dijkstra; VF2 K₃≡K₃; Hopcroft-Karp 3×3 perfect; Hungarian = brute-force 4×4; triangle K₄=4; PTX × 6 SM
- [x] `benches/graphalg_ops.rs` — Criterion: 7 PTX kernels × all SM + Dijkstra / BFS / PageRank / Floyd-Warshall / Edmonds-Karp / Louvain algo benches

---

## Vol.60: Numerical Analysis Primitives [COMPLETE]

### oxicuda-numeric (~71 files, ~7,260 SLoC, 212 tests)
- [x] `error.rs` — `NumericError` enum (14 variants: NotConverged, RootNotBracketed, ShapeMismatch, InvalidParameter, NumericalInstability, UnsupportedSmVersion, IndexOutOfBounds, DimensionMismatch, EmptyInput, DegreeTooHigh, OutOfDomain, …) + `NumericResult<T>`
- [x] `handle.rs` — `SmVersion`, `LcgRng` (MMIX LCG, bit-32 bool, Box-Muller), `NumericHandle`
- [x] `ptx_kernels.rs` — 7 kernels × 6 SM versions: `horner_eval`, `rk4_stage`, `bisection_step`, `gauss_quad_accumulate`, `spline_eval`, `central_diff`, `bessel_recurrence` (string concatenation only)
- [x] `root/{bisection,newton,secant,brent,halley,aberth_all_roots}.rs` — Bisection O(log(1/ε)); Newton quadratic conv; secant superlinear (~1.618); Brent (bisection + secant + IQI); Halley cubic; Aberth-Ehrlich simultaneous all-roots for polynomial
- [x] `quadrature/{romberg,gauss_legendre,gauss_hermite,gauss_laguerre,gauss_chebyshev,clenshaw_curtis,adaptive_simpson,gauss_kronrod}.rs` — Romberg with Richardson; Gauss nodes via Golub-Welsch (Jacobi eigh); Chebyshev closed-form; Clenshaw-Curtis with DFT weights; adaptive Simpson with `15·ε` criterion; G7-K15 pair with `|G7-K15|^1.5` error
- [x] `special/{bessel_jy,bessel_ik,airy,lambert_w,hypergeometric_2f1,elliptic_ke,zeta,dilogarithm,ei,polygamma}.rs` — Bessel J/Y/I/K via Miller's algorithm + Wronskian normalisation; Airy Ai/Bi power series + asymptotic; Lambert W₀/W₋₁ Halley; ₂F₁ Taylor + transformations; elliptic K/E via AGM; ζ via Euler-Maclaurin + functional eq; Li₂ series + transformations; exponential integral Ei; digamma/trigamma recurrence + asymptotic
- [x] `ode/{explicit_euler,heun,rk4,dopri5,bdf12,rosenbrock_w,imex_euler}.rs` — Forward Euler; Heun RK2; classical RK4 (k1,k2,k3,k4); Dormand-Prince RK45 7-stage embedded with PI controller adaptive step; BDF1/BDF2 with Newton inner; Rosenbrock-W linearly implicit; IMEX explicit + implicit split
- [x] `poly/{durand_kerner,jenkins_traub,companion_matrix_eigvals,horner_eval,deflate}.rs` — Simultaneous all-roots Durand-Kerner / Weierstrass; Jenkins-Traub three-stage RPOLY (no shift, fixed shift, variable shift); companion matrix Hessenberg + QR shifts; Horner + derivative; synthetic-division deflate
- [x] `diff/{central_difference,richardson_extrapolation,complex_step}.rs` — Central diff O(h²); Richardson combines D(h) + D(h/2) for O(h⁴); complex-step `Im(f(x+ih))/h` avoids subtractive cancellation
- [x] `interp/{linear,cubic_spline,akima,pchip,lagrange,hermite,barycentric}.rs` — Linear; natural + clamped cubic spline (Thomas tridiagonal for M); Akima 5-point slopes; Fritsch-Carlson PCHIP (monotone); Lagrange O(n²); Hermite with values + derivatives; barycentric Lagrange `wⱼ = 1/Π_{i≠j}(xⱼ-xᵢ)`
- [x] `cubature/{monte_carlo,quasi_monte_carlo_sobol,tensor_product_gauss,genz_malik}.rs` — Monte Carlo O(1/√N) with stderr; Sobol/Halton low-discrepancy (verified VdC base-2 in dim 1); tensor-product Gauss; Genz-Malik 1980 adaptive degree-7 fully-symmetric basic rule
- [x] `linalg/{jacobi_eig,qr_givens,lu_decomp,householder_qr}.rs` — Private cyclic Jacobi eigh; QR via Givens; LU with partial pivoting; Householder QR
- [x] `metrics/metrics.rs` — Absolute/relative error, max-norm, residual norm, 2×2 condition number
- [x] `e2e_tests.rs` — 38 cross-module tests: bisection cos→π/2 to 1e-10; Newton x³-2 → 2^(1/3) <15 iter; Brent sin→π to 1e-12; Romberg 1/(1+x²)→π/4; Gauss-Legendre n=5 exact on x⁹; adaptive Simpson 1/√x [0,1]; bessel_j0(0)=1, j0(2.4048…)≈0; airy_ai(0)=1/(3^(2/3)Γ(2/3)); lambert_w₀(e)=1; elliptic_k(0)=π/2; RK4 exp-decay 1e-4; DOPRI5 harmonic energy conservation; cubic spline through (x,x³) at 1.5 ≈ 3.375; PCHIP monotone; Durand-Kerner roots of (x-1)(x-2)(x-3); Sobol VdC; PTX × 6 SM
- [x] `benches/numeric_ops.rs` — Criterion: 7 PTX kernels × all SM + Bessel / Gauss-Kronrod / DOPRI5 / Bowyer-Watson / cubic-spline / Aberth algo benches

---

## Vol.61: 2D Computational Geometry [COMPLETE]

### oxicuda-geom2d (~74 files, ~7,066 SLoC, 190 tests)
## Stubs to implement (added 2026-06-12 by /cooljapan-stub-check)

- [x] `oxicuda-ptx`: `crates/oxicuda-ptx/src/analysis/kernel_fusion.rs:273` — replace placeholder estimated_speedup (hardcoded 1.0) with real cost-model estimate in kernel fusion analysis
  - Priority: P2 | Scope: small | Hint: none
- [x] `oxicuda-survival`: `crates/oxicuda-survival/src/cox/gradient_boost.rs:483` — replace placeholder leaf-node value (0.0) with real fitted leaf value in GB tree
  - Priority: P2 | Scope: small | Hint: none
- [x] `oxicuda-evol`: `crates/oxicuda-evol/src/neuroevolution/es_hyperneat.rs:428` — replace placeholder hidden coordinates `[(0.0, 0.0)]` with real hidden node placement algorithm
  - Priority: P2 | Scope: medium | Hint: none
- [x] `oxicuda-driver`: `crates/oxicuda-driver/src/stream_ordered_alloc.rs:30` — implement real stream-ordered allocation (placeholder stream handle in example)
  - Priority: P2 | Scope: medium | Hint: none

- [x] `error.rs` — `Geom2dError` enum (DegeneratePolygon, NotEnoughPoints, InvalidParameter, NumericalInstability, UnsupportedSmVersion, IndexOutOfBounds, DimensionMismatch, EmptyInput, NotConvex, NotSimplePolygon, ParallelSegments, …) + `Geom2dResult<T>`
- [x] `handle.rs` — `SmVersion`, `LcgRng` (MMIX LCG, bit-32 bool, Box-Muller), `Geom2dHandle`
- [x] `ptx_kernels.rs` — 7 kernels × 6 SM versions: `orientation_test`, `cross_product`, `point_in_aabb`, `segment_intersection`, `convex_hull_step`, `kd_tree_traverse`, `polygon_area` (string concatenation only)
- [x] `primitives/{point,vector,line,segment,ray,circle,aabb,polygon}.rs` — Standard 2D primitives with Add/Sub/Mul ops, `dot`, `cross = a.x*b.y - a.y*b.x`, `norm`, `norm_sq`, `distance`, `rotate(θ)`, `reflect`
- [x] `predicate/{orientation,in_circle,dot_cross,robust_signs}.rs` — `o(a,b,c) = (b-a) × (c-a)` with CCW/CW/collinear via ε; in-circle 4-point determinant; robust signs with ε
- [x] `intersection/{segment_segment,line_line,segment_polygon,circle_segment,circle_circle}.rs` — Parametric segment-segment with collinear-overlap; line-line via 2×2 solve; circle-segment quadratic; circle-circle radical line
- [x] `containment/{point_in_polygon_winding,point_in_polygon_ray_cast,point_in_convex_polygon,point_in_circle}.rs` — Winding-number signed crossings (robust for non-convex); ray-casting; O(log n) convex via binary search; in-circle norm²-test
- [x] `hull/{graham_scan,andrew_monotone_chain,quickhull,jarvis_march,chans_algorithm}.rs` — Graham polar-angle sort + sweep with orientation tests; Andrew x-sort + upper/lower chains; QuickHull D&C; Jarvis O(nh) gift-wrap; Chan O(n log h) optimal
- [x] `triangulation/{ear_clipping,bowyer_watson_delaunay,constrained_delaunay}.rs` — O(n²) ear clipping; Bowyer-Watson incremental Delaunay (find conflicting triangles → retriangulate cavity); constrained Delaunay with flip-and-restore
- [x] `voronoi/{fortune_sweepline,voronoi_from_delaunay}.rs` — Fortune sweepline with parabolic beach line + site/circle events; dual-graph Voronoi from Delaunay circumcenters
- [x] `clipping/{sutherland_hodgman,weiler_atherton,line_clip_cohen_sutherland,liang_barsky}.rs` — Sutherland-Hodgman convex-clip; Weiler-Atherton non-convex; Cohen-Sutherland bit-coded line-AABB; Liang-Barsky parametric line clip
- [x] `polygon_ops/{area_shoelace,centroid,perimeter,convexity_test,polygon_offset,minkowski_sum}.rs` — Shoelace `A = (1/2)|Σ (xᵢyᵢ₊₁ - xᵢ₊₁yᵢ)|`; centroid `(1/6A)Σ ...`; perimeter; convexity by sign-consistency; edge-shift offset; convex Minkowski sum via angle merge
- [x] `closest_pair/{divide_conquer,brute_force}.rs` — O(n log n) divide-and-conquer + O(n²) baseline
- [x] `enclosing/{welzl_smallest_circle,axis_aligned_bbox,rotating_calipers_diameter,rotating_calipers_width}.rs` — Welzl expected-O(n) smallest enclosing circle; AABB; rotating-calipers diameter + width on convex hull
- [x] `sweepline/bentley_ottmann.rs` — O((n+k) log n) all-segment intersection reporting
- [x] `point_location/{slab_method,trapezoidal_map}.rs` — Slab vertical decomposition with binary search; Seidel randomized trapezoidal map
- [x] `index/{kd_tree_2d,rtree_2d,quadtree}.rs` — 2D KD-tree alternating x/y splits with kNN + radius search; R-tree with STR bulk-loading; quadtree recursive 4-way subdivision
- [x] `metrics/metrics.rs` — Euclidean, Manhattan, Chebyshev distance; angle between vectors; signed area
- [x] `e2e_tests.rs` — 20 cross-module tests: CCW orientation `(0,0),(1,0),(0,1)`; unit square contains (0.5,0.5); convex hull of 5-point set returns 4 corners; Graham/Andrew/QuickHull agree; segment intersection at (1,1); shoelace = 1; centroid = (0.5,0.5); Welzl radius = √2/2; Bowyer-Watson degenerate-collinear error; 4-pt → 2 triangles; closest pair = 1; Sutherland-Hodgman; Fortune perpendicular bisector; Bentley-Ottmann 4 crossings; KD-tree kNN ≡ brute force; PTX × 6 SM
- [x] `benches/geom2d_ops.rs` — Criterion: 7 PTX kernels × all SM + convex-hull / Delaunay / Welzl / KD-tree / segment-intersection algo benches
