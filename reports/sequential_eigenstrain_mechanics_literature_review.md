# Mechanics and Numerical Methods for Sequential Eigenstrain Shell Forming

## Research Frame

- **engineering_domain**: Sequential eigenstrain-driven forming of thin bilayer/non-Euclidean shell structures for prescribed double-curvature shapes
- **method_family_of_interest**: Finite inelastic kinematics, incompatible shell mechanics, nonlinear continuation and stability tracking, and uncertainty-aware multifidelity surrogate modeling
- **open_problem**: Define and solve a sequential toolpath evolution law whose equilibria are repeatable and converge jointly in mesh size, nonlinear tolerance, and load/path increment, while remaining useful as a low-fidelity design model
- **baseline_approach**: Per-path prescribed target-metric increments followed by local HLBFGS minimization of discrete bilayer shell energy
- **data_regime**: Many inexpensive but numerically noisy low-fidelity simulations; sparse or presently unavailable process-resolved and experimental calibration data
- **scope_notes**: [Distinguish prescribed natural-metric models from process-resolved plasticity, treat 20-path double-curvature loading, assess constitutive accumulation and equilibrium branch selection separately]
- **failure_modes**: [Run-to-run branch variability, mesh-dependent equilibria, tolerance-dependent basin selection, undefined physical meaning of eigenstrain accumulation, unstable error propagation across paths, unverified Hessian/stability classification, surrogates learning solver artifacts]
- **scope_boundaries**: [No claim that one metric recurrence is universally correct, no full thermo-metallurgical process model unless literature shows it is necessary, no surrogate accuracy claim before solver discrepancy is quantified]
- **origin_layer**: 5
- **validation_status**: frame-converged

## Executive conclusion

The current code is a defensible **prescribed incompatible-metric shell model**, but it is not yet a validated model of sequential plastic forming. Those are different claims.

The main weakness is not simply that HLBFGS is the wrong optimizer. Three layers are currently entangled:

1. **Constitutive evolution:** what a tool pass changes in the material;
2. **Equilibrium-path evolution:** which equilibrium or dynamic transition follows that change;
3. **Numerical approximation:** mesh, load increment, nonlinear residual, and linear algebra errors.

Changing optimizers cannot determine the missing constitutive law. Conversely, calibrating a better eigenstrain law will not fix uncontrolled equilibrium branch switching. The recommended program is therefore:

- retain the natural-metric shell as a low-cost structural response model;
- identify its eigenstrain evolution law from one-pass and overlapping-pass evidence;
- replace one-shot per-path minimization with adaptive substepping plus a controlled equilibrium corrector;
- track projected tangent-stiffness eigenvalues and branches;
- model snap-through dynamically only when the quasistatic branch loses stability;
- treat the resulting simulator as low fidelity until joint mesh, load-step, solver, and experimental discrepancies are quantified.

## What the repository currently implements

### Evidence

- `src/libshell/GrowthHelper.hpp:796-819` constructs a material-plane growth tensor and offers multiplicative, recursive-linearized, and reference-additive updates of each face's target first fundamental form.
- `src/simulations/Sim_Bilayer_Growth.cpp:1808-1817` holds the explicit reference second fundamental form fixed. Differential evolution of the top and bottom target metrics still supplies the bilayer's effective curvature drive through stretching–bending coupling; adding a second independent curvature update without re-derivation could double-count that incompatibility.
- `src/simulations/Sim_Bilayer_Growth.cpp:1684-1699` applies fixed scheduled cycles. It has solve cadence, but no adaptive eigenstrain-step rejection or bisection.
- `src/simulations/Sim_Bilayer_Growth.cpp:1824-1848` calls HLBFGS after selected cycles and records the termination report. More critically, `lastMinimization.converged()` is called without a calibrated gradient threshold, while `Sim.hpp:207-213` permits line-search termination codes 1 and 4 to count as converged. Correct residual-based acceptance is a prerequisite to adaptive retry; otherwise a stalled state can be accepted and seed the next pass.
- `src/simulations/Sim_Bilayer_Growth.cpp:1135-1141` permits only HLBFGS. The statement that Newton methods enter a “wrong physical basin” is a code comment/assertion, not experimental identification of the physical branch.
- `src/libshell/TinyADHessian_Bilayer.hpp:255-316` already provides exact-gradient and matrix-free Hessian-vector products for the discrete energy.
- `src/simulations/Sim_Bilayer_Growth.cpp:2077-2136` uses a fixed-count power iteration without rigid-mode projection or an eigenpair residual. It is a diagnostic, not a rigorous stability certificate.
- The corrected ablation report found a `51.95 µm` maximum all-pairs repeatability floor and unresolved mesh dependence (`reports/sequence_ablation_v2_report.md`).

### Inference

Errors can propagate through 20 paths because every path changes both the material state and the next nonlinear initial condition. Near a soft mode or bifurcation, a perturbation that is geometrically small after path 5 can determine the branch reached after path 20. Error growth need not be linear; it can remain small until a stability boundary and then cause an order-one mode switch.

### Unknown

The repository does not establish whether a real wheel/peening/tool pass changes an in-plane natural metric, a through-thickness plastic-strain tensor, residual stress, natural curvature, or some combination. That must be supplied by mechanics and calibration, not selected by whichever recurrence gives the smoothest simulation.

## 1. Repeatability for realistic 20-path double curvature

The previous repeated-run study used repeated instances of one toolpath topology. It did not test 20 distinct paths designed to build double curvature. A realistic repeatability study needs a state-by-state divergence analysis.

For run replicas `r` and path index `k`, record

\[
D_k^{(r,s)}=d_{SE(3)}(X_k^{(r)},X_k^{(s)}),\qquad
M_k^{(r,s)}=\|\bar a_k^{(r)}-\bar a_k^{(s)}\|,
\]

plus residual norm, projected minimum tangent eigenvalue, energy, and branch label. If `M_k=0` but `D_k` grows, the divergence is equilibrium/solver driven. If both grow, mapping, constitutive update, or parallel accumulation is involved. If divergence jumps when the projected minimum eigenvalue approaches zero, it is branch sensitivity rather than ordinary accumulation error.

A valid experiment should include:

1. five to ten identical runs of a 20-distinct-path double-curvature schedule;
2. deterministic single-thread and fixed multi-thread controls;
3. path-increment subdivisions `1, 2, 4, 8` per tool pass;
4. at least three meshes with identical physical path rasterization rules;
5. per-path projected Hessian spectrum and mode-shape correlation;
6. controlled geometric imperfections, because a perfectly symmetric shell may not define a unique realized branch.

Repeatability is then a trajectory property, not only a final-shape statistic.

## 2. What “closer to the true optimum” should mean

A finer mesh and tighter tolerance are not automatically truth. They reduce selected error sources, but can also resolve a different branch or expose discretization defects. The target is a converged **solution path**, not the deepest energy found by any run.

For an observable `Q`, seek a plateau under joint refinement:

\[
Q(h,\Delta\lambda,\tau,\eta_{eig})
\rightarrow Q^*,
\]

where `h` is mesh size, `Δλ` is the eigenstrain/load increment, `τ` is the nonlinear residual tolerance, and `η_eig` is the eigensolver residual. Refinement should also preserve branch identity, assessed by aligned mode shapes and equilibrium-path continuity.

A useful error budget is

\[
\epsilon_Q \lesssim
\epsilon_{mesh}+
\epsilon_{step}+
\epsilon_{nonlinear}+
\epsilon_{stability}+
\epsilon_{constitutive}.
\]

The present study only partially measured the first and third terms. The following sequence is more discriminating:

1. **Freeze constitutive state:** compare equilibrium algorithms on exactly the same target metric.
2. **Solver convergence:** require force residual, accepted-step history, energy consistency, and projected second-order diagnostic.
3. **Load-step convergence:** subdivide every path increment until the same branch and final observables stabilize.
4. **Mesh convergence:** refine geometry and toolpath integration together; compare common material-space observables, not vertex indices.
5. **Constitutive convergence/calibration:** only after numerical errors are below experimental/model discrepancy.

There may be no unique “true optimum.” A quasistatic shell follows a locally stable equilibrium branch until loss of stability. A global energy minimum can be mechanically inaccessible because of an energy barrier. Arc-length methods can trace stable and unstable equilibria; they do not decide the physically realized jump. Dynamic evolution, damping, imperfections, and experiments decide that transition.

## 3. How to choose an eigenstrain accumulation law

### Two independent modeling axes

The first distinction is **state representation**:

| State representation | Retains | Loses or approximates |
|---|---|---|
| Infinitesimal eigenstrain tensor | linear residual-strain field | finite rotations, multiplicative history |
| Finite plastic/growth distortion `Fᵖ` or `Fᵍ` | anisotropic stretch, plastic spin, ordered composition | requires a constitutive evolution law and often 3D/thickness resolution |
| Natural metric `Cᵖ=(Fᵖ)ᵀFᵖ` or shell `ā` | intrinsic relaxed lengths; frame-indifferent elastic energy | plastic spin and some history can be quotiented out |
| Top/bottom shell metrics and natural curvature | reduced through-thickness incompatibility | detailed stress/strain distribution and contact history |
| Residual-stress field | direct elastic driving stress | transfer between meshes/configurations is difficult; needs equilibrium compatibility |

The second distinction is the **evolution law**:

| Evolution law | Physical interpretation | Evidence needed |
|---|---|---|
| Prescribed additive increment | linearized or reference-measure superposition | increments sufficiently small and calibrated in that measure/frame |
| Recursive linearized increment | first-order update about current natural state | subdivision convergence and controlled finite-rotation error |
| Multiplicative composition | ordered finite distortion increments | increment defined relative to the current intermediate/natural configuration |
| Return mapping / flow rule | stress-dependent elastoplastic evolution | yield function, hardening, unloading, material parameters |
| Empirical operator learned from passes | process-to-state map | one-pass, overlap, reversal, saturation, and transfer data |

An additive update of an intrinsic target metric does not by itself violate spatial rigid-motion objectivity: the shell energy still compares intrinsic forms. Its problem is different—it may be the wrong approximation to the process, strain measure, or intermediate configuration. Similarly, multiplicative congruence preserves positive definiteness but is not automatically physical. It assumes that each prescribed increment acts in the particular natural/material frame and composition order implemented in the code.

### Discriminating calibration program

Do not compare recurrence formulas after giving them the same numerical `g`; that does not ensure the same one-pass physical action. Instead:

1. Calibrate each candidate law to the **same isolated one-pass coupon response**.
2. Predict, without recalibration:
   - two overlapping passes;
   - reversed pass order;
   - `0°/45°` noncommuting orientations;
   - repeated passes to saturation;
   - unload/reload and opposite-direction passes;
   - thickness and width changes.
3. Compare not only final shape but residual strain/stress, curvature, and springback.
4. Add a matched-final-metric control. If the shape still differs, the difference comes from equilibrium history/branch selection rather than final natural metric.
5. Use path subdivision to determine whether each discrete recurrence approximates a continuous rate law.

If prestress changes the next pass response, none of the three present metric-only recurrences is sufficient. A stress-coupled internal-variable law or process-resolved reference model is then required.

## 4. Numerical architecture: change the algorithm, not merely the optimizer name

The fair comparison has two layers.

### Outer path evolution

- fixed prescribed substeps;
- adaptive substeps with rejection/bisection;
- pseudo-arclength continuation for a scalarized pass amplitude;
- dynamic or pseudo-transient transition after loss of stability.

### Inner corrector

- current HLBFGS energy minimization;
- damped Newton on equilibrium residuals with safeguards;
- trust-region Newton-CG using Hessian-vector products;
- adaptive cubic regularization;
- deflated Newton as an offline branch-discovery tool, not the production physical trajectory.

The first implementation experiment should be factorial: outer fixed/adaptive stepping crossed with inner HLBFGS/trust-region Newton-CG. Hold the energy, mesh, target-metric trajectory, starting state, residual target, evaluation budget, and deterministic perturbation fixed. Compare success rate, evaluations, trajectory repeatability, load-step convergence, and projected tangent inertia. This identifies whether improvement comes from smaller path increments or from the local corrector.

Trust-region Newton-CG is attractive because the repository already has `hessianVectorProduct`. It can recognize negative curvature in its local quadratic model. It does **not** guarantee the physically correct branch or a global minimum. Arc length can traverse limit points and unstable branches. It does **not** supply transition dynamics. These are complementary layers.

A robust proposed algorithm is:

1. parameterize each path update by `λ∈[0,1]`;
2. predict the next equilibrium using the previous path tangent;
3. for candidate locally stable states, compare HLBFGS with trust-region Newton-CG as the corrector;
4. accept only if a calibrated projected residual, energy/model agreement, and state continuity pass; raw line-search termination is insufficient;
5. otherwise reduce `Δλ` and retry;
6. compute the smallest eigenpairs of the tangent projected off rigid modes using a residual-controlled symmetric eigensolver;
7. near a zero eigenvalue, reduce step and record the mode;
8. after loss of stability, choose the task explicitly: use a safeguarded Newton solve of the augmented equilibrium residual plus arc-length constraint to trace an unstable branch for analysis, or use a calibrated damped dynamic transition for physical prediction;
9. continue after the transition with a persistent branch identifier.

## 5. Can this be a low-fidelity surrogate model?

Yes, but not as a deterministic oracle for final vertex coordinates.

The present model can be useful because it is cheap, has meaningful mechanics structure, and exposes a spatial natural-metric field. Its current discrepancy is too large and too structured to hide inside ordinary interpolation error. A surrogate trained now would mix:

- intended design dependence;
- mesh/toolpath rasterization error;
- nonlinear stopping error;
- branch choice;
- missing constitutive physics;
- thread-level numerical variation.

### Minimum evidence gates

Before using simulation outputs as low-fidelity training labels:

1. **Trajectory repeatability:** 20-path repeated runs with per-step divergence and branch labels.
2. **Numerical convergence:** mesh, path-substep, nonlinear, and eigensolver residual studies.
3. **Stable output representation:** rigid alignment plus intrinsic fields or modal/curvature descriptors; never raw unaligned vertices.
4. **Branch separation:** do not average opposing buckling branches in a single Gaussian likelihood.
5. **Replicated designs:** estimate input-dependent numerical variance.
6. **High-fidelity anchors:** process-resolved simulations or measured shapes/residual strains at selected designs.
7. **Out-of-sample validation:** include path schedules, growth levels, and geometries not used in calibration.

### Surrogate families must be benchmarked, not assumed

The current data do not establish heteroscedastic Gaussian noise, an input-partitionable discontinuity, coexisting outputs at identical inputs, or useful low/high-fidelity correlation. Compare these baselines under identical train/test splits and cost:

- deterministic GP or ensemble with a fitted nugget;
- homoscedastic and heteroscedastic probabilistic models, justified by replicated within-branch data;
- a branch classifier followed by branch-conditional regressors;
- treed GP and mixture-of-experts models, noting that a treed GP alone cannot represent coexisting branches at the same input;
- low-fidelity-only, high-fidelity-only/no-transfer, linear discrepancy, and nonlinear multifidelity fusion.

Select using branch-classification accuracy, negative log predictive density/calibration, conditional shape error against physical references, and improvement per unit cost. Use identifiable discrepancy-aware calibration so constitutive parameters and a flexible discrepancy process do not explain the same error.

At deployment, the first model should estimate `p(branch | ordered toolpath/history, fidelity controls, imperfection descriptor)`. Separate conditional regressors may then consume an observed or sampled branch label and return aligned modal/curvature features, conditional geometry, and numerical/model uncertainty. For inverse design, acquisition should penalize regions where branch identity or fidelity transfer is unresolved.

## Per-Paper Summary Blocks

### [Efrati, Sharon, and Kupferman, 2009] — Elastic theory of unconstrained non-Euclidean plates

**Assumptions** — Thin elastic body with a prescribed reference metric; incompatible metric may not admit a stress-free Euclidean embedding.

**Outputs / Contributions** — Derives a covariant elastic plate theory separating stretching and bending around a non-Euclidean reference geometry.

**Gaps / Limitations** — The target metric is prescribed; no manufacturing-process evolution law is supplied.

**Cross-Field Transfer Potential** — Strong foundation for the repository's structural response model, but not evidence for how wheel passes should accumulate eigenstrain. [DOI](https://doi.org/10.1016/j.jmps.2008.12.004)

### [van Rees, Vouga, and Mahadevan, 2017] — Growth patterns for shape-shifting elastic bilayers

**Assumptions** — Thin elastic bilayers with programmed differential growth represented by target forms.

**Outputs / Contributions** — Develops the discrete bilayer shell framework directly related to this codebase and demonstrates programmed shape generation.

**Gaps / Limitations** — Does not model stress-dependent sequential plastic forming or physical snap dynamics.

**Cross-Field Transfer Potential** — Retain as the low-cost shell-response layer; add process evolution and controlled path algorithms around it. [DOI](https://doi.org/10.1073/pnas.1709025114)

### [Lewicka, Mahadevan, and Pakzad, 2011] — Models for elastic shells with incompatible strains

**Assumptions** — Thin-body asymptotics and prescribed incompatible prestrain under defined energy scalings.

**Outputs / Contributions** — Establishes rigorous limiting plate models and connects incompatibility to curvature-driving source terms.

**Gaps / Limitations** — Asymptotic theory does not identify toolpath-induced plastic evolution parameters.

**Cross-Field Transfer Potential** — Guides which thickness/mesh regimes a reduced shell should reproduce and what convergence claims are meaningful. [DOI](https://doi.org/10.1098/rspa.2010.0138)

### [Pezzulla, Smith, Nardinocchi, and Holmes, 2016] — Geometry and mechanics of thin growing bilayers

**Assumptions** — Thin bilayers under controlled differential growth and near-isometric response.

**Outputs / Contributions** — Relates natural curvature to realized curvature and explains cylinder-like symmetry breaking.

**Gaps / Limitations** — Idealized growth and geometry; not a sequential metal-forming constitutive law.

**Cross-Field Transfer Potential** — Provides analytic benchmarks for curvature and branch behavior that the numerical shell should pass before complex paths. [DOI](https://doi.org/10.1039/C6SM00246C)

### [Rodriguez, Hoger, and McCulloch, 1994] — Stress-dependent finite growth

**Assumptions** — Multiplicative decomposition of elastic and growth distortions with a constitutive growth evolution.

**Outputs / Contributions** — Demonstrates how finite inelastic distortion and stress-dependent evolution can be separated from elastic response.

**Gaps / Limitations** — Developed for biological growth, not rolling or peening plasticity.

**Cross-Field Transfer Potential** — Conceptually useful for defining a process state and evolution law; parameters cannot be transferred directly. [DOI](https://doi.org/10.1016/0021-9290(94)90021-3)

### [Lee, 1969] — Elastic-plastic deformation at finite strains

**Assumptions** — Finite elastoplastic kinematics with an intermediate configuration and multiplicative decomposition.

**Outputs / Contributions** — Foundational representation `F=FᵉFᵖ` for separating elastic and plastic distortions at finite strain.

**Gaps / Limitations** — Kinematics alone do not determine the yield surface, hardening, flow, plastic spin, or integration algorithm.

**Cross-Field Transfer Potential** — Shows what information a full finite-plasticity model could retain beyond a 2D metric. It does not prove the code's particular multiplicative metric recurrence. [DOI](https://doi.org/10.1115/1.3564580)

### [Lubarda, 2004] — Constitutive theories based on the multiplicative decomposition

**Assumptions** — Finite inelasticity represented using elastic and inelastic distortions under constitutive restrictions.

**Outputs / Contributions** — Reviews the mechanical meaning, advantages, and ambiguities of multiplicative decompositions and intermediate configurations.

**Gaps / Limitations** — Multiple constitutive choices remain; a decomposition is not a process calibration.

**Cross-Field Transfer Potential** — Useful for checking objectivity and identifying when the natural metric is sufficient versus when plastic spin/history matters. [DOI](https://doi.org/10.1115/1.1591000)

### [Goodbrake, Goriely, and Yavari, 2021] — Geometric nonlinearization of incompatible elasticity

**Assumptions** — Inelastic distortions and natural geometries formulated geometrically.

**Outputs / Contributions** — Clarifies compatibility and the conditions under which a relaxed intermediate geometry can be embedded.

**Gaps / Limitations** — Does not prescribe tool/material-specific evolution.

**Cross-Field Transfer Potential** — Supports treating incompatible metrics as genuine material state variables while exposing what compatibility checks mean. [DOI](https://doi.org/10.1098/rspa.2020.0462)

### [Riks, 1979] — Incremental solution of snapping and buckling problems

**Assumptions** — Smooth quasistatic equilibrium manifold with a load parameter and consistent tangent.

**Outputs / Contributions** — Arc-length continuation through limit points where load control fails.

**Gaps / Limitations** — Does not choose a physical branch after dynamic instability; multi-parameter toolpaths require scalarization or generalized continuation.

**Cross-Field Transfer Potential** — Treat each eigenstrain pass amplitude as a continuation parameter and measure branch continuity instead of relying on one-shot minimization. [DOI](https://doi.org/10.1016/0020-7683(79)90081-7)

### [Crisfield, 1981] — Arc-length procedure for snap-through

**Assumptions** — Geometrically nonlinear finite-element equilibrium corrected by Newton iterations.

**Outputs / Contributions** — Practical cylindrical arc-length constraint and root-selection procedure.

**Gaps / Limitations** — Step size and branch switching still require care; unstable traced states are not automatically physical trajectories.

**Cross-Field Transfer Potential** — A practical outer continuation candidate for subdivided path amplitudes. [DOI](https://doi.org/10.1016/0045-7949(81)90108-5)

### [Steihaug, 1983] — Conjugate gradients and trust regions

**Assumptions** — Twice-differentiable objective with Hessian-vector products.

**Outputs / Contributions** — Truncated-CG trust-region subproblem solver that detects negative curvature without Hessian factorization.

**Gaps / Limitations** — Finds a local optimization trajectory; it does not identify which branch a physical shell follows.

**Cross-Field Transfer Potential** — High implementation leverage because exact matrix-free Hessian-vector products already exist in the repository. [DOI](https://doi.org/10.1137/0720042)

### [Cartis, Gould, and Toint, 2011] — Adaptive cubic regularisation

**Assumptions** — Smooth nonconvex objective with controlled Hessian variation.

**Outputs / Contributions** — Cubically regularized local models with strong worst-case convergence properties toward approximate second-order stationarity.

**Gaps / Limitations** — Krylov subproblem implementation and preconditioning remain substantial engineering work; physical branch selection remains external.

**Cross-Field Transfer Potential** — A second inner-corrector candidate if trust-region Newton-CG is not robust enough near very flat shell modes. [DOI](https://doi.org/10.1007/s10107-009-0286-5)

### [Stein, Wagner, and Wriggers, 1990] — Nonlinear shell stability and branch switching

**Assumptions** — Nonlinear shell equilibrium with tangent-stiffness monitoring and identifiable critical modes.

**Outputs / Contributions** — Stability detection and controlled eigenmode perturbation for branch exploration.

**Gaps / Limitations** — Multiple or clustered modes and imperfection sensitivity complicate branch switching.

**Cross-Field Transfer Potential** — Supports projected eigenvalue tracking and controlled branch audits, replacing heuristic perturbation claims. [DOI](https://doi.org/10.1007/BF01113447)

### [Riks, Rankin, and Brogan, 1996] — Mode jumping in thin-walled shells

**Assumptions** — Thin shells with mode interaction and loss of static stability; dynamic evolution used during jumps.

**Outputs / Contributions** — Couples static path analysis with transient response to study mode jumping.

**Gaps / Limitations** — Requires mass, damping, time integration, and calibration; the destination can depend on imperfections and damping.

**Cross-Field Transfer Potential** — Relevant if experiments show abrupt path-to-path snaps that a purely quasistatic solver cannot reproduce. [DOI](https://doi.org/10.1016/0045-7825(95)00970-1)

### [Farrell, Birkisson, and Funke, 2015] — Deflation for distinct nonlinear solutions

**Assumptions** — Differentiable nonlinear residual and multiple isolated solutions.

**Outputs / Contributions** — Deflation prevents reconvergence to known solutions and discovers coexisting equilibria.

**Gaps / Limitations** — Found solutions need separate stability assessment; enumeration does not establish physical accessibility.

**Cross-Field Transfer Potential** — Use offline to map competing shell branches at selected path stages and train branch-aware surrogates. [DOI](https://doi.org/10.1137/140984798)

### [Peherstorfer, Willcox, and Gunzburger, 2018] — Multifidelity methods survey

**Assumptions** — Multiple information sources with useful correlation and known cost/accuracy tradeoffs.

**Outputs / Contributions** — Unifies multifidelity estimation, optimization, and reduced modeling; explains when low fidelity reduces cost.

**Gaps / Limitations** — Correlation can collapse across bifurcations; application-specific discrepancy structure is required.

**Cross-Field Transfer Potential** — Frames the natural-metric shell as low fidelity only after measuring correlation with experiments or process-resolved models. [DOI](https://doi.org/10.1137/16M1082469)

### [Perdikaris et al., 2017] — Nonlinear information fusion for multifidelity modeling

**Assumptions** — High-fidelity response has a learnable nonlinear relationship with low-fidelity response plus discrepancy.

**Outputs / Contributions** — Nonlinear autoregressive multifidelity Gaussian-process architecture.

**Gaps / Limitations** — A single smooth mapping can fail across unlabelled branches or discontinuities.

**Cross-Field Transfer Potential** — Suitable after branch conditioning, using sparse experiments to correct systematic natural-metric-shell discrepancy. [DOI](https://doi.org/10.1098/rspa.2016.0751)

### [Binois, Gramacy, and Ludkovski, 2018] — Practical heteroscedastic Gaussian-process modeling

**Assumptions** — Replicated observations reveal input-dependent noise, with locally smooth mean and variance processes.

**Outputs / Contributions** — Scalable joint inference of response and heteroscedastic noise.

**Gaps / Limitations** — Noise models do not by themselves represent multiple distinct equilibrium branches.

**Cross-Field Transfer Potential** — Directly useful for replicated solver trajectories after branch labels are separated. [DOI](https://doi.org/10.1080/10618600.2018.1458625)

### [Gramacy and Lee, 2008] — Bayesian treed Gaussian processes

**Assumptions** — Input space can be partitioned into regions with simpler local response models.

**Outputs / Contributions** — Nonstationary surrogate with data-driven partitions and local GPs.

**Gaps / Limitations** — Branch overlap at the same input requires a latent branch variable, not only input partitioning.

**Cross-Field Transfer Potential** — Useful for discontinuous changes across stability boundaries when branch identity is observable. [DOI](https://doi.org/10.1198/016214508000000689)

### [Rasmussen and Ghahramani, 2002] — Infinite mixtures of Gaussian-process experts

**Assumptions** — Data arise from multiple latent regimes that can be represented by separate GP experts.

**Outputs / Contributions** — Bayesian mixture of local GP models with nonparametric allocation.

**Gaps / Limitations** — Label switching and sparse data complicate physical interpretation; mechanics-informed branch descriptors are still needed.

**Cross-Field Transfer Potential** — Prevents averaging incompatible shell branches into a nonphysical mean geometry. [Official paper](https://proceedings.neurips.cc/paper/2055-infinite-mixtures-of-gaussian-process-experts)

### [Tuo and Wu, 2015] — Efficient calibration for imperfect computer models

**Assumptions** — Reality differs systematically from the simulator and calibration must address parameter/discrepancy confounding.

**Outputs / Contributions** — Defines calibration through an `L2` projection and analyzes identifiability and convergence.

**Gaps / Limitations** — Requires experimental observations and an appropriate discrepancy space; branch-changing outputs complicate distance definitions.

**Cross-Field Transfer Potential** — Prevents eigenstrain parameters from absorbing all model-form error when calibrating the shell to measured panels. [DOI](https://doi.org/10.1214/15-AOS1314)

### [Bessa and Pellegrino, 2018] — Bayesian machine learning for shell buckling

**Assumptions** — Shell response depends strongly on uncertain imperfections and can be learned probabilistically from simulations/data.

**Outputs / Contributions** — Demonstrates probabilistic machine-learning treatment of imperfection-sensitive post-buckling shell response.

**Gaps / Limitations** — Structural and loading setting differs from sequential eigenstrain forming; transfer must be tested.

**Cross-Field Transfer Potential** — Supports treating imperfections and branch response probabilistically rather than demanding one deterministic final geometry. [DOI](https://doi.org/10.1016/j.ijsolstr.2018.01.035)

## Gap Analysis

### Addressed by retrieved literature

- **Run-to-run branch variability:** stability tracking, imperfection modeling, deflation, mixtures of experts.
- **Tolerance-dependent basin selection:** adaptive continuation and safeguarded second-order correctors.
- **Undefined accumulation:** finite-inelastic kinematics and natural-metric theories clarify possible state variables.
- **Surrogate averaging of branches:** treed and mixture models provide appropriate statistical structure.

### Partially addressed

- **Mesh-dependent equilibria:** incompatible-shell asymptotics constrain the target theory, but this code still needs a discretization-specific convergence and locking study.
- **Error propagation across 20 paths:** continuation theory explains amplification near critical points, but direct evidence requires the proposed 20-path trajectory experiment.

### Unaddressed gaps

- No retrieved paper establishes the correct process-to-eigenstrain evolution law for this exact toolpath/material/thickness combination.
- No current repository evidence validates the natural-metric state against residual-strain or residual-stress measurements.
- No existing test identifies physical post-buckling transitions because mass, damping, rate, geometric imperfections, and experimental trajectories are absent.
- The current Hessian diagnostic lacks rigid-mode deflation and residual-controlled eigenpairs.

## Research Agenda

### Direction — highest tractable

**Apply/improve adaptive continuation and projected stability tracking from nonlinear structural mechanics to solve non-repeatable 20-path equilibrium evolution in sequential eigenstrain-driven shell forming.**

- Add scalar path-amplitude substeps, rejection/bisection, projected residual-controlled eigenpairs, and persistent branch labels.
- Cross fixed/adaptive stepping with HLBFGS/trust-region Newton-CG rather than replacing both layers simultaneously.
- First experiment: five replicated 20-distinct-path double-curvature schedules on three meshes and four substep levels. Success means trajectory differences stabilize below discretization error and no unconverged stage seeds the next path.

### Direction — medium

**Apply/improve finite-inelastic internal-variable calibration from computational plasticity and morphoelasticity to solve ambiguous eigenstrain accumulation in sequential eigenstrain-driven shell forming.**

- Compare state representations and evolution laws as separate axes.
- Calibrate each candidate to identical one-pass data; validate on overlap, reversal, noncommuting orientation, saturation, and springback without recalibration.
- Escalate from natural metric to stress-coupled return mapping only if prestress/history tests falsify metric-only evolution.

### Direction — stretch-exploratory

**Apply/improve branch-conditioned nonlinear multifidelity Gaussian processes from uncertainty quantification to solve surrogate learning under solver noise and bifurcation in sequential eigenstrain-driven shell forming.**

- A classifier consumes ordered toolpath/history, fidelity controls, and an imperfection descriptor to estimate branch probabilities; the unknown branch label is not an input.
- Separate branch-conditional regressors consume an observed or sampled branch and predict aligned modal/curvature features, conditional geometry, and heteroscedastic uncertainty when replicates support it.
- Begin only after trajectory repeatability and numerical convergence gates; benchmark against no-transfer baselines and anchor with sparse experiments or process-resolved simulations.

## Recommended order of work

1. Implement measurement before replacement: per-stage residuals, accepted steps, projected eigenpairs, branch-mode correlation, and failure retry.
2. Add path-amplitude subdivision and adaptive rejection around the current HLBFGS solver.
3. Run the 20-distinct-path trajectory repeatability/convergence experiment.
4. Implement trust-region Newton-CG as a second inner corrector using the existing Hessian-vector product; compare factorially.
5. Perform one-pass/overlap/reversal/saturation calibration to identify the state/evolution law.
6. Add dynamic transitions only if measured forming trajectories show snap events that quasistatic continuation cannot represent.
7. Build a branch-conditioned low-fidelity surrogate only after the preceding uncertainty budget is available.

## References

1. Efrati, E., Sharon, E., & Kupferman, R. (2009). *Elastic theory of unconstrained non-Euclidean plates*. Journal of the Mechanics and Physics of Solids, 57, 762–775. https://doi.org/10.1016/j.jmps.2008.12.004
2. van Rees, W. M., Vouga, E., & Mahadevan, L. (2017). *Growth patterns for shape-shifting elastic bilayers*. PNAS, 114, 11597–11604. https://doi.org/10.1073/pnas.1709025114
3. Lewicka, M., Mahadevan, L., & Pakzad, M. R. (2011). *The Föppl-von Kármán equations for plates with incompatible strains*. Proceedings of the Royal Society A, 467, 402–426. https://doi.org/10.1098/rspa.2010.0138
4. Pezzulla, M., Smith, G. P., Nardinocchi, P., & Holmes, D. P. (2016). *Geometry and mechanics of thin growing bilayers*. Soft Matter, 12, 4435–4442. https://doi.org/10.1039/C6SM00246C
5. Rodriguez, E. K., Hoger, A., & McCulloch, A. D. (1994). *Stress-dependent finite growth in soft elastic tissues*. Journal of Biomechanics, 27, 455–467. https://doi.org/10.1016/0021-9290(94)90021-3
6. Lee, E. H. (1969). *Elastic-plastic deformation at finite strains*. Journal of Applied Mechanics, 36, 1–6. https://doi.org/10.1115/1.3564580
7. Lubarda, V. A. (2004). *Constitutive theories based on the multiplicative decomposition of deformation gradient: Thermoelasticity, elastoplasticity, and biomechanics*. Applied Mechanics Reviews, 57, 95–108. https://doi.org/10.1115/1.1591000
8. Goodbrake, C., Goriely, A., & Yavari, A. (2021). *The mathematical foundations of anelasticity: existence of smooth global intermediate configurations*. Proceedings of the Royal Society A, 477, 20200462. https://doi.org/10.1098/rspa.2020.0462
9. Riks, E. (1979). *An incremental approach to the solution of snapping and buckling problems*. International Journal of Solids and Structures, 15, 529–551. https://doi.org/10.1016/0020-7683(79)90081-7
10. Crisfield, M. A. (1981). *A fast incremental/iterative solution procedure that handles snap-through*. Computers & Structures, 13, 55–62. https://doi.org/10.1016/0045-7949(81)90108-5
11. Steihaug, T. (1983). *The conjugate gradient method and trust regions in large scale optimization*. SIAM Journal on Numerical Analysis, 20, 626–637. https://doi.org/10.1137/0720042
12. Cartis, C., Gould, N. I. M., & Toint, P. L. (2011). *Adaptive cubic regularisation methods for unconstrained optimization. Part I*. Mathematical Programming, 127, 245–295. https://doi.org/10.1007/s10107-009-0286-5
13. Stein, E., Wagner, W., & Wriggers, P. (1990). *Nonlinear stability-analysis of shell and contact-problems including branch-switching*. Computational Mechanics, 5, 428–446. https://doi.org/10.1007/BF01113447
14. Riks, E., Rankin, C. C., & Brogan, F. A. (1996). *On the solution of mode jumping phenomena in thin-walled shell structures*. Computer Methods in Applied Mechanics and Engineering, 136, 59–92. https://doi.org/10.1016/0045-7825(95)00970-1
15. Farrell, P. E., Birkisson, Á., & Funke, S. W. (2015). *Deflation techniques for finding distinct solutions of nonlinear partial differential equations*. SIAM Journal on Scientific Computing, 37, A2026–A2045. https://doi.org/10.1137/140984798
16. Peherstorfer, B., Willcox, K., & Gunzburger, M. (2018). *Survey of multifidelity methods in uncertainty propagation, inference, and optimization*. SIAM Review, 60, 550–591. https://doi.org/10.1137/16M1082469
17. Perdikaris, P., Raissi, M., Damianou, A., Lawrence, N. D., & Karniadakis, G. E. (2017). *Nonlinear information fusion algorithms for data-efficient multi-fidelity modelling*. Proceedings of the Royal Society A, 473, 20160751. https://doi.org/10.1098/rspa.2016.0751
18. Binois, M., Gramacy, R. B., & Ludkovski, M. (2018). *Practical heteroscedastic Gaussian process modeling for large simulation experiments*. Journal of Computational and Graphical Statistics, 27, 808–821. https://doi.org/10.1080/10618600.2018.1458625
19. Gramacy, R. B., & Lee, H. K. H. (2008). *Bayesian treed Gaussian process models with an application to computer modeling*. Journal of the American Statistical Association, 103, 1119–1130. https://doi.org/10.1198/016214508000000689
20. Rasmussen, C. E., & Ghahramani, Z. (2002). *Infinite mixtures of Gaussian process experts*. NeurIPS 14, 881–888. https://proceedings.neurips.cc/paper/2055-infinite-mixtures-of-gaussian-process-experts
21. Tuo, R., & Wu, C. F. J. (2015). *Efficient calibration for imperfect computer models*. Annals of Statistics, 43, 2331–2352. https://doi.org/10.1214/15-AOS1314
22. Bessa, M. A., & Pellegrino, S. (2018). *Design of ultra-thin shell structures in the stochastic post-buckling range using Bayesian machine learning and optimization*. International Journal of Solids and Structures, 139–140, 174–188. https://doi.org/10.1016/j.ijsolstr.2018.01.035
