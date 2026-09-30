# Trajectory-Aware Multi-Fidelity Active Sampling

**Date:** 2026-08-20<br>
**Status:** Research-direction and benchmark survey<br>
**Application:** English-wheel and related incremental forming processes<br>
**Method scope:** General controlled dynamical systems with history, multiple fidelities,
field-valued outputs, and expensive observations

## 1. Executive summary

The sampling policy can be a research contribution rather than a preprocessing choice.
The core problem is broader than English-wheel forming:

> Given a controlled, history-dependent dynamical system, select the next input
> trajectory, process parameters, and fidelity level that most efficiently improve a
> calibrated field-to-field surrogate of the real system.

The essential query is

\[
q=(\mathcal T,\boldsymbol\theta,\ell),
\]

where \(\mathcal T\) is a dynamically feasible control trajectory,
\(\boldsymbol\theta\) contains material, geometry, and process parameters, and
\(\ell\) chooses a fidelity such as a coarse model, refined simulation, or physical
experiment.

No single benchmark needs to reproduce English-wheel physics. A defensible method paper
should use a ladder:

1. analytic multi-fidelity functions to verify cost-aware acquisition;
2. hysteretic or constitutive dynamical systems to verify memory and order effects;
3. active-learning and controlled-PDE suites to verify spatiotemporal operator learning;
4. public mechanics/forming data to test field outputs and simulation-to-real transfer;
5. the project solver and new experiments as the final application benchmark.

The strongest immediately usable dynamical-system benchmarks are
[AL4PDE](https://github.com/dmusekamp/al4pde),
[PDEControlGym](https://github.com/lukebhan/PDEControlGym), and
[Constitutive OED](https://github.com/lcao11/constitutive_oed). The strongest public
forming resources are [DB4ISF](https://zenodo.org/records/10000815), the
[TINY incremental-forming dataset](https://zenodo.org/records/17917675), and the paired
[DDACS](https://doi.org/10.18419/DARUS-4801)-
[RDDAC](https://doi.org/10.18419/DARUS-5589) simulation/experiment datasets.

## 2. Problem formulation

### 2.1 Physical characteristics

The target process combines:

- **History dependence.** Plastic strain, hardening, residual stress, springback, and
  evolving contact geometry depend on the preceding loading path.
- **Moving localized actuation.** A small contact patch produces non-local deformation
  over a much larger sheet.
- **Function-valued control.** The input is a trajectory
  \(\mathcal T(t)=\{\mathbf x(\tau),F_N(\tau),v(\tau),\ldots\}_{\tau=0}^t\), not an
  unconstrained vector.
- **Field-valued response.** Desired outputs include geometry, displacement, thinning,
  strain, curvature, residual stress, or uncertainty fields.
- **Multiple fidelities.** Different solvers and experiments have different bias, cost,
  resolution, noise, and availability.

### 2.2 Sequential operator model

A history-aware surrogate should learn a transition operator

\[
s_{k+1}=\mathcal G_\phi(s_k,a_k,\boldsymbol\theta),
\]

where \(s_k\) contains observable fields and internal-history summaries and \(a_k\) is
the next path segment or process action. A one-shot map from the accumulated final
eigenstrain to final geometry is insufficient when different application orders can
produce different states.

A real-system model can be decomposed as

\[
\mathcal G_R
=
\mathcal G_{LF}(\cdot;\boldsymbol\theta_{cal})
+\Delta_{LF\rightarrow R}(\cdot)
+\epsilon_R,
\]

where \(\boldsymbol\theta_{cal}\) are calibratable physical parameters,
\(\Delta_{LF\rightarrow R}\) is model discrepancy, and \(\epsilon_R\) is experimental
noise. Parameter calibration and discrepancy must not be silently conflated.

### 2.3 Active learning versus adaptive sampling

- **Adaptive sampling** is the complete sequential workflow: initialize, acquire,
  retrain, recalibrate, validate, and stop.
- **Active learning** is the acquisition mechanism that chooses informative new labels
  using epistemic uncertainty, expected error reduction, information gain, diversity,
  or physics residuals.
- **Bayesian optimal experimental design** chooses experiments to learn parameters or
  predictive quantities of interest.
- **Multi-fidelity acquisition** additionally decides which information source should
  answer the query.

For this project, active learning is one component of a broader adaptive, multi-fidelity
experimental-design loop.

## 3. Literature map and research gap

| Literature family | Representative work | Contribution | Missing element for this project |
|---|---|---|---|
| Operator learning | [DeepONet](https://doi.org/10.1038/s42256-021-00302-5), [FNO](https://openreview.net/forum?id=c8P9NQVtmnO) | Function-to-function surrogate models | No inherent history, fidelity policy, or calibrated epistemic acquisition |
| Continuous-history models | [Neural CDE](https://proceedings.neurips.cc/paper/2020/hash/4a5876b450b45371f6cfe5047ac8cd45-Abstract.html), [history-aware neural operator](https://arxiv.org/abs/2506.10352) | Variable-length, irregular, or autoregressive histories | Training trajectories are ordinarily fixed in advance |
| Path-dependent plasticity | [Deep learning predicts path-dependent plasticity](https://doi.org/10.1073/pnas.1911815116), [Constitutive OED](https://arxiv.org/abs/2603.12365) | Learned constitutive memory; experimental design for history-dependent laws | Generally material-point or prescribed test design, not moving field actuation |
| Active operator learning | [Extreme-event discovery](https://doi.org/10.1038/s43588-022-00376-0), [AL4PDE](https://openreview.net/forum?id=x4ZmQaumRg), [active operator UQ](https://doi.org/10.1016/j.jcp.2026.114791) | Acquisition in high-dimensional input-function spaces | Mostly single-fidelity PDE instances; no real calibration or order-pair design |
| Multi-resolution active learning | [MRA-FNO](https://proceedings.mlr.press/v238/li24k.html) | Jointly selects input functions and numerical resolution by utility per cost | Resolution fidelity is simpler than shell/solid/experiment discrepancy |
| Multi-fidelity fusion | [Nonlinear information fusion](https://doi.org/10.1098/rspa.2016.0751), [multi-fidelity DeepONet](https://doi.org/10.1016/j.jcp.2023.112462) | Learns nonlinear LF-HF relationships | Usually assumes the query locations are already selected |
| Cost-aware multi-fidelity BO | [MF-MES](https://proceedings.mlr.press/v119/takeno20a.html), [continuous-fidelity KG](https://openreview.net/forum?id=SknC0bW0-) | Selects an input-fidelity pair by information per cost | Usually finite-dimensional inputs and scalar objectives |
| Calibration and discrepancy | [Kennedy-O'Hagan](https://doi.org/10.1111/1467-9868.00294), [calibration identifiability](https://doi.org/10.1088/0266-5611/30/11/114007) | Separates calibration, discrepancy, and observation noise | Not a trajectory-aware active operator framework |
| Trajectory-aware forming | [AI surrogate for incremental forming](https://doi.org/10.1016/j.rineng.2026.109927), [DB4ISF](https://doi.org/10.1007/s00170-024-14014-8) | Encodes full toolpaths and intermediate/final deformation | Uses preselected datasets rather than cost-aware adaptive acquisition |

In the literature surveyed here, no framework jointly addresses:

1. continuous constrained control trajectories;
2. non-commuting history and reversal effects;
3. field-valued operator outputs;
4. heterogeneous simulation and experimental fidelities;
5. epistemic, cost-aware, batch-diverse acquisition; and
6. simultaneous physical calibration and model-discrepancy learning.

The proposed novelty is therefore:

> **Trajectory-aware, cost-sensitive, multi-fidelity active learning for
> history-dependent dynamical operators, with paired order probes and explicit
> simulation-to-reality calibration.**

This is an inference from the surveyed literature, not a claim that no related work can
exist outside the sources reviewed.

## 4. Candidate acquisition formulation

At iteration \(n\), choose

\[
(\mathcal T^*,\boldsymbol\theta^*,\ell^*)
=
\arg\max_{\mathcal T,\boldsymbol\theta,\ell}
\frac{
\mathbb E[\Delta R_R]
+\lambda_{cal}I(\boldsymbol\theta_{cal};Y_\ell)
+\lambda_{ord}U_{ord}
+\lambda_{phys}U_{phys}
}{c_\ell(\mathcal T)},
\]

subject to admissible trajectory, load, velocity, acceleration, workspace, and safety
constraints.

- \(\Delta R_R\): expected reduction in real-fidelity predictive risk;
- \(I(\boldsymbol\theta_{cal};Y_\ell)\): information gained about calibration
  parameters;
- \(U_{ord}\): uncertainty about reversal or ordering effects;
- \(U_{phys}\): stability, conservation, residual, or failure-boundary utility;
- \(c_\ell\): measured wall-time, compute, or experimental cost.

For batch acquisition, select a diverse subset in a trajectory embedding rather than the
top \(B\) uncertainty values. Candidate embeddings can use spline coefficients, causal
transformer features, neural-CDE states, or path signatures. Epistemic uncertainty should
be estimated separately from measurement and process noise.

Paired non-commutativity probes should be protected in the design:

\[
\mathcal T,\quad \operatorname{reverse}(\mathcal T),\quad
A\circ B,\quad B\circ A.
\]

A normalized order-effect statistic is

\[
C(A,B)=
\frac{\|G(A\circ B)-G(B\circ A)\|_W}{\sigma_{repeat}},
\]

where \(\sigma_{repeat}\) is experimental repeatability noise.

## 5. Benchmark selection requirements

A benchmark does not need to be a forming process. It should expose several of the
following:

- a trajectory, time-dependent forcing, or control input;
- latent memory, hysteresis, irreversible evolution, or long temporal dependence;
- an oracle that can generate new labels, not only a static table;
- one or more controllable fidelity knobs;
- field-valued or trajectory-valued outputs;
- known evaluation cost;
- uncertainty, noise, or a simulation-to-real gap;
- constraints on valid controls;
- a license permitting research reuse.

Static datasets permit **pool-based** active learning. They cannot by themselves test
continuous query synthesis. Generator-backed benchmarks are therefore more important for
developing the acquisition method.

## 6. Benchmark inventory

### 6.1 Acquisition unit tests

| Benchmark | Query and fidelity | Availability | Recommended use | Limitation |
|---|---|---|---|---|
| [EmuKit multi-fidelity functions](https://github.com/EmuKit/emukit) | Forrester, Branin, Currin, Hartmann, Park, and Borehole with analytic fidelity levels | Public, Apache-2.0 | Verify acquisition, cost normalization, asynchronous batches, and stopping | No dynamics or field outputs |
| [BoTorch multi-fidelity Hartmann](https://botorch.org/docs/next/tutorials/multi_fidelity_bo) | Design point plus continuous fidelity coordinate | Public, MIT | Verify continuous-fidelity knowledge-gradient implementation | No history |
| Self-generated hysteresis operator | Piecewise control path; LF/HF integration or constitutive law | Must be generated | Verify path reversal, paired order tests, and uncertainty calibration with exact ground truth | Requires us to define and publish the protocol |

### 6.2 History-dependent dynamical systems

| Benchmark | Dynamics and input | Fidelity opportunities | Availability | Fit |
|---|---|---|---|---|
| [Constitutive OED](https://github.com/lcao11/constitutive_oed) | Time-dependent experiment design for linear viscoelastic and nonlinear history-dependent laws | Linear/nonlinear model, surrogate/FEM, design resolution, Monte Carlo budget | Public code; repository includes a license | **Very high:** directly tests whether sampling discovers informative histories |
| [Nonlinear Bouc-Wen Frame](https://github.com/KosVla/NonlinearBoucWenFrameBenchmark) | Cyclic/stochastic excitation and hysteretic structural response | Model size, damage complexity, integration resolution, excitation family | Public simulator and standardized datasets, Apache-2.0 | **High:** cheap, clear memory and reversal effects |
| [Z3ST](https://github.com/giozu/z3st) | Thermo-mechanical histories with plasticity, creep, contact, and fracture | Dimension, strain model, mesh, and constitutive physics | Public code, Apache-2.0 | High-fidelity generator for path-dependent mechanics |
| [rvesimulator](https://github.com/bessagroup/rvesimulator) | User-defined loading histories for material RVEs | Mesh, microstructure, constitutive model, and DoE size | Public code, MIT | Useful for multiscale constitutive discovery; computationally heavier |
| Self-generated J2 suite | Load-unload-reload, cyclic, ratcheting, dwell, and \(A\to B\) versus \(B\to A\) paths | 1D/plane stress/3D, coarse/fine increments, isotropic/kinematic hardening | Must be generated | **Recommended first mechanics benchmark:** fully controlled and interpretable |

### 6.3 Active operator learning and controlled PDEs

| Benchmark | Query/control | Oracle and fidelity | Availability | Fit |
|---|---|---|---|---|
| [AL4PDE](https://github.com/dmusekamp/al4pde) | Initial conditions and PDE parameters for Burgers, Kuramoto-Sivashinsky, conservation laws, and 2D Navier-Stokes | Solver-in-the-loop; extensible task, simulator, model, and batch acquisition interfaces | Public benchmark code | **Best initial benchmark for the acquisition method.** It already compares random, uncertainty, and feature-diverse batch selection |
| [PDEControlGym](https://github.com/lukebhan/PDEControlGym) | Time-dependent boundary control for transport/Burgers-type, reaction-diffusion, and 2D Navier-Stokes systems | Bundled numerical environments; fidelity can be added through grid/time resolution or alternate solvers | Public, CC BY-NC-SA 4.0 | **Best control-trajectory analogue.** Requires extending its benchmark from control return to surrogate-information acquisition |
| [MRA-FNO](https://proceedings.mlr.press/v238/li24k.html) | Selects input functions and numerical resolution for Burgers, Darcy, nonlinear diffusion, and Navier-Stokes | Two or three spatial resolutions; utility-cost acquisition | Paper and public implementation referenced by the authors | **Best direct multi-resolution active-operator baseline** |
| [PDEBench](https://github.com/pdebench/PDEBench) | Initial/boundary conditions and PDE parameters | Data-generation code plus multiple resolutions/parameters | Public code and data, MIT except where stated | Strong general operator baseline; active learning must be added |
| [The Well](https://github.com/PolymathicAI/the_well) | Short state histories for 16 spatiotemporal systems | Static datasets at substantial scale; some systems admit downsampled fidelity | Public, BSD-3-Clause | Good rollout stress test, but mostly pool-based and up to 15 TB |
| [PDEArena](https://github.com/pdearena/pdearena) | Parametric spatiotemporal PDE rollouts | Data generators and benchmark models | Public, MIT | Useful for temporal generalization; less direct fidelity support |

### 6.4 Mechanics and forming datasets

| Benchmark | Contents | Availability | Suitable task | Limitation |
|---|---|---|---|---|
| [Mechanical MNIST Multi-Fidelity](https://open.bu.edu/handle/2144/41357) | Coarse/fine deformation fields for heterogeneous materials | Public data; CC BY-SA 4.0, MIT code | Validate coarse-to-fine field fusion and distribution-shift robustness | Static loading rather than continuous control |
| [TINY incremental forming](https://zenodo.org/records/17917675) | 1,975 FEM forming trajectories with different die geometries; intermediate/final deformation and springback data; 10.3 GB | Public download; source page states copyright but no open reuse license | Pool-based trajectory-aware forming surrogate and offline active-learning replay | Single simulation fidelity, fixed resolution, no experiments; clarify reuse terms before redistribution |
| [DB4ISF](https://zenodo.org/records/10000815) | 76 physical incremental-forming experiments with CAD, toolpaths, robot programs, digitization, and per-path-point deviation | Public, 3.7 GB, CC BY 4.0 | Experimental path encoder, real-data uncertainty, and path-family holdouts | Small and not paired to a public simulation bank |
| [DDACS](https://doi.org/10.18419/DARUS-4801) | 32,466 two-stage deep-drawing/cutting simulations with mesh fields and springback | Public, about 640 GB, CC BY 4.0; MIT loader | Large simulation pretraining and field-output acquisition | Process parameters rather than freely synthesized trajectories |
| [RDDAC](https://doi.org/10.18419/DARUS-5589) | About 9,000 repeated real deep-drawing/cutting experiments with forces, thickness, oil, and 3D scans | Public, about 81 GB, CC BY 4.0; paired conceptually with DDACS | **Best public simulation-to-real calibration benchmark** | Real and simulated geometries/parameterizations require careful alignment |
| [NUMISHEET 2014 Benchmark 3](https://figshare.com/articles/conference_contribution/Benchmark_3_incremental_sheet_forming/20913343) | SPIF cone: strain, punch load, deformed profile, and springback | Public problem description; all rights reserved | Historical forming validation case | Not a modern queryable dataset |
| [NUMISHEET 2025](https://numisheet2025.com/benchmarks/) | Constitutive calibration and blind forming validation with force and DIC strain outputs | Public benchmark results and open-access reports; data availability varies by component | Calibration protocol and material-model identifiability | Not trajectory sampling |
| [EXACT](https://zenodo.org/records/6874577) | Material tests and aluminum cup-drawing benchmark data | Public Zenodo record | Calibration and validation workflow | No moving control path |
| [AM-Bench](https://www.nist.gov/ambench/am-bench-data-and-challenge-problems-0) | Moving laser scan patterns with thermal, melt-pool, cooling, and microstructure measurements | Public NIST challenge data; verify component-specific terms | Strong analogue for moving localized actuation and path ordering | Thermal phase transformation rather than mechanical contact |

## 7. Project-specific English-wheel benchmark

The current `zigzag_sequence` implementation is already an oracle for ordered cycles:

- JSON toolpaths execute in array order;
- target metrics and face hit counts persist across cycles;
- hardening is recomputed from previous qualifying hits;
- one minimization occurs after each complete toolpath repeat;
- there is no intermediate mechanical relaxation between individual strips or hits.

Relevant implementation locations are
`src/simulations/Sim_Bilayer_Growth.cpp:920`, `:1560`, `:1676`, and `:1751`,
with hardening defined in `src/libshell/ZigZagSequenceGrowth.hpp:580`.

This creates natural fidelity levels:

| Fidelity | Proposed oracle |
|---|---|
| LF-0 | Whole-layer or accumulated-eigenstrain equilibrium |
| LF-1 | Current whole-toolpath-repeat shell relaxation |
| MF | Segment-wise or strip-wise shell relaxation |
| HF | 3D elasto-plastic contact FEA with refined local mesh |
| Real | Wheel experiment with profilometry/DIC and process measurements |

Required paired order experiments include:

- forward versus reversed traversal;
- \(A\to B\) versus \(B\to A\);
- identical total nominal eigenstrain with different temporal grouping;
- repeated overlapping passes;
- top/bottom or up/down alternation;
- short localized paths versus full-length paths.

The absence of a public dataset combining wheel contact, reversed paths, paired
simulation fidelities, and physical measurements makes this benchmark itself a potential
dataset contribution.

## 8. Recommended minimum viable benchmark suite

Do not start with every dataset. The following suite isolates the claims efficiently:

1. **EmuKit Forrester or BoTorch Hartmann:** prove the cost-aware fidelity policy is
   implemented correctly.
2. **Self-generated J2 or Bouc-Wen paths:** prove the sampler detects history,
   reversal, and non-commutativity.
3. **AL4PDE:** compare the new acquisition against an established active neural-PDE
   protocol.
4. **PDEControlGym:** test continuous constrained controls rather than only initial
   conditions.
5. **Mechanical MNIST Multi-Fidelity or MRA-FNO:** test field-valued coarse/fine fusion.
6. **DDACS-RDDAC:** test simulation-to-reality calibration and discrepancy learning.
7. **Project shell/experiment hierarchy:** demonstrate the complete method on the target
   application.

DB4ISF and TINY are valuable secondary forming tests, but neither alone provides the
queryable paired-fidelity oracle needed to validate the full proposed method.

## 9. Baselines and evaluation protocol

### 9.1 Sampling baselines

- random or Sobol acquisition with fixed fidelity ratios;
- uncertainty-only acquisition;
- feature-distance or core-set diversity only;
- uncertainty plus diversity;
- physics-residual acquisition;
- conventional multi-fidelity BO in a reduced trajectory space;
- the proposed trajectory-and-fidelity acquisition;
- an HF-only upper-cost baseline and, where possible, an oracle acquisition bound.

### 9.2 Surrogate baselines

- recurrent MLP/GRU/LSTM;
- causal transformer or neural CDE;
- FNO/DeepONet without explicit memory;
- autoregressive/history-aware operator;
- single-fidelity and multi-fidelity residual variants;
- ensemble and calibrated-UQ variants.

### 9.3 Metrics

The horizontal axis should be cumulative acquisition cost, not sample count.

- integrated and worst-case field error versus cost;
- rollout error versus horizon;
- error on held-out trajectory families;
- negative log likelihood and empirical interval coverage;
- calibration error for epistemic uncertainty;
- order-effect detection power relative to repeatability noise;
- posterior contraction and bias of physical parameters;
- LF-HF-real discrepancy prediction;
- downstream control or toolpath optimization regret;
- wall time, parallel efficiency, and number of real experiments.

Data splits must occur at the trajectory or path-family level. Splitting adjacent states
from the same rollout across training and test sets would leak history and overstate
generalization.

## 10. Research questions

1. Which path representation preserves traversal order while remaining stable under
   temporal re-discretization?
2. Can the acquisition policy choose both trajectory and fidelity more efficiently than
   fixed-ratio or uncertainty-only sampling?
3. When does an LF model provide useful information despite systematic physical bias?
4. Can paired reversal and path-swap queries identify non-commutative dynamics with fewer
   samples than generic uncertainty acquisition?
5. Can calibration parameters and model discrepancy be identified separately from sparse
   experiments?
6. Do benefits transfer from generic dynamical benchmarks to forming and downstream
   toolpath optimization?

## 11. Remaining verification and risks

- Verify license terms before redistributing any TINY, NUMISHEET, or NIST data; public
  download does not always imply an open redistribution license.
- Static public datasets only support pool-based replay unless a simulator is available.
- Resolution changes are an easier fidelity relationship than changing governing physics;
  conclusions from MRA-FNO or Mechanical MNIST must not be overstated.
- Deep ensembles can remain overconfident out of distribution; coverage calibration and
  held-out path families are required.
- Calibration and discrepancy can be non-identifiable without informative experiments and
  defensible priors.
- Multistability requires branch-aware outputs or a probabilistic response model; a
  deterministic mean predictor can average incompatible equilibria.
