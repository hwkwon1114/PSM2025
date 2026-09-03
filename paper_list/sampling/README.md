# Sampling and kernel-learning literature map

This library separates five ideas that answer different questions. Treating
them as interchangeable is the main conceptual failure to avoid.

1. **Pilot design without response labels** chooses the first simulations using
   space-filling or distribution-matching criteria.
2. **Response-grounded information design** learns a covariance or likelihood
   from observations and then selects informative new experiments.
3. **Functional-input and learned kernels** model relationships between entire
   input fields and, in some cases, entire output fields.
4. **Active neural-operator learning** uses surrogate uncertainty, physics
   residuals, or coresets to choose additional function-valued training cases.
5. **Temporal/path kernels** apply only when order changes a physical internal
   state. They are not primary methods for the present order-insensitive regime.

Downloaded PDFs are grouped in the numbered directories beside this file. They
come from open-access publisher, proceedings, author, or arXiv copies and are
ignored by Git. `manifest.tsv` records the source URL for every local file.

## Big idea 1: obtain the pilot before a response covariance exists

These methods do not claim to know which samples are physically informative.
They provide defensible initial coverage from which a response model can be
learned.

| Paper | Main idea | What it assumes | Use here |
|---|---|---|---|
| [Qian, Ai & Wu — Nested space-filling designs](01_pilot_design/qian-2009-nested-space-filling-designs.pdf) | Construct nested designs for sequential or multifidelity campaigns. | A finite-dimensional parameterization and a geometric coverage criterion. | Strong template for budgets 25, 50, 100, 200, and 500. |
| [Lin et al. — Flexible computer-experiment designs](01_pilot_design/lin-2010-flexible-computer-experiment-designs.pdf) | Recursive orthogonal Latin-hypercube construction. | Coordinate-space stratification is meaningful. | Appropriate for calibration parameters and as a pilot baseline. |
| [Mak & Joseph — Support points](01_pilot_design/mak-joseph-2018-support-points.pdf) | Match a finite design to a target input distribution using energy distance. | The desired input distribution is known or chosen. | Attractive for spatial treatment fields when a population distribution can be specified. |

**Interpretation:** this is where the current project must start. Before pilot
responses exist, call the criterion coverage, not mutual information about the
forming response.

## Big idea 2: information design requires a learned response model

| Paper | Main idea | What it assumes | Use here |
|---|---|---|---|
| [Krause, Singh & Guestrin — GP sensor placement](02_gp_information/krause-2008-near-optimal-sensor-placement.pdf) | Select observations by mutual information under a GP covariance. | Repeated observations or an initial deployment have produced a credible covariance. | Canonical reference after a curvature-response pilot, not before it. |
| [Krause et al. — Robust submodular selection](02_gp_information/krause-2008-robust-submodular-observation-selection.pdf) | Optimize against several possible objectives/models. | Each candidate model supplies a meaningful submodular objective. | Useful when covariance or calibration uncertainty remains large. |
| [Kleinegesse et al. — MI neural estimation](02_gp_information/kleinegesse-2020-mi-neural-estimation-design.pdf) | Estimate information gain for implicit simulator models. | Simulated joint samples can train the MI estimator. | Relevant if an explicit GP likelihood becomes too restrictive. |
| [Jakkala & Akella — Sparse-GP sensor placement](02_gp_information/jakkala-2023-sparse-gp-sensor-placement.pdf) | Scalable continuous/discrete sensor placement using sparse GPs. | A response GP remains an adequate uncertainty model. | Useful computational reference if the candidate pool grows. |
| [Wei, Iyer & Bilmes — Submodular data selection](02_gp_information/wei-2015-submodular-data-selection-active-learning.pdf) | Combine uncertainty filtering with diverse subset selection. | The surrogate uncertainty is informative enough to filter candidates. | A practical batch-acquisition pattern after operator training begins. |
| [Tremblay et al. — DPP coresets](02_gp_information/tremblay-2019-dpp-coresets.pdf) | Select diverse coresets with determinantal point processes. | The similarity kernel is meaningful for the task. | A diversity mechanism, not by itself a physical information measure. |

**Key distinction:** Krause et al. estimated covariance from real repeated
temperature/precipitation observations. An arbitrary RBF kernel on treatment
geometry does not reproduce that methodology.

## Big idea 3: learn on function-valued inputs and outputs

| Paper | Main idea | What it assumes | Use here |
|---|---|---|---|
| [Muehlenstaedt et al. — GP with functional inputs](03_functional_kernels/muehlenstaedt-2014-functional-input-gp.pdf) | Build GP covariance from norms between input functions. | The selected functional norm tracks response similarity. | Direct baseline for spatial curvature/treatment inputs. |
| [Sung et al. — Functional-input GP](03_functional_kernels/sung-2022-functional-input-gp.pdf) | Kernels designed for functional inputs in a physics problem. | Functional inputs can be compared through the proposed kernel structure. | One of the closest GP precedents for treatment-field inputs. |
| [Damiano et al. — Functional relevance determination](03_functional_kernels/damiano-2022-functional-input-relevance.pdf) | Learn which regions/components of a functional input matter. | Enough labeled responses exist to identify relevance. | Valuable for discovering which treatment-field regions affect curvature. |
| [Wilson et al. — Deep kernel learning](03_functional_kernels/wilson-2016-deep-kernel-learning.pdf) | Learn a neural representation inside a GP covariance by marginal likelihood. | GP likelihood and uncertainty remain usable after representation learning. | Main learned-kernel candidate after the pilot. |
| [Ober et al. — Promises and pitfalls of DKL](03_functional_kernels/ober-2021-promises-pitfalls-deep-kernel-learning.pdf) | Diagnose failure modes and overconfidence in deep kernels. | — | Required caution alongside Wilson et al.; do not trust marginal likelihood alone. |
| [Kohl et al. — Learned similarity for simulations](03_functional_kernels/kohl-2020-learned-similarity-numerical-simulations.pdf) | Learn task-relevant distances between numerical simulation fields. | Training pairs or supervision define useful similarity. | Strong bridge from raw curvature fields to response-aware sampling metrics. |
| [Kadri et al. — Operator-valued kernels](03_functional_kernels/kadri-2016-operator-valued-kernels-functional-response.pdf) | Kernel regression when inputs and responses are functions. | Operator-valued kernel computation is tractable. | Classical field-to-field alternative and uncertainty reference. |
| [Nelsen & Stuart — Operator learning with random features](03_functional_kernels/nelsen-stuart-2024-operator-learning-random-features.pdf) | Random-feature approximation of function-space operators. | A suitable operator-valued kernel/prior can be specified. | Scalable kernel/operator baseline against neural operators. |

Additional archived papers cover distribution inputs, guided kernel learning,
multiple operator-valued kernels, and function-valued RKHS methods. See the
manifest rather than treating every archived paper as required reading.

## Big idea 4: select training functions for a neural operator

| Paper | Main idea | What it assumes | Use here |
|---|---|---|---|
| [Kovachki et al. — Neural Operator](04_operator_active_learning/kovachki-2023-neural-operator.pdf) | General framework for maps between function spaces. | A representative training distribution exists. | Defines the final surrogate objective, not the acquisition rule. |
| [Li et al. — Multi-resolution active FNO](04_operator_active_learning/li-2024-multiresolution-active-fno.pdf) | Choose both input functions and resolution using utility per cost. | Ensemble uncertainty ranks useful cases and resolution is a fidelity axis. | Closest mature active-learning workflow for our simulation hierarchy. |
| [Musekamp et al. — AL4PDE](04_operator_active_learning/musekamp-2024-active-learning-neural-pde-solvers.pdf) | Benchmark pool-based acquisition for neural PDE solvers. | A candidate pool and retraining loop are available. | Useful evaluation framework for comparing acquisition policies. |
| [Pickering et al. — Active neural operators for extremes](04_operator_active_learning/pickering-2022-active-learning-neural-operators-extremes.pdf) | Bayesian experimental design with an ensemble of deep neural operators. | The goal emphasizes rare/extreme responses and ensemble uncertainty is useful. | Relevant if high-curvature or buckling tails matter disproportionately. |
| [Subedi & Tewari — Active operator data collection](04_operator_active_learning/subedi-2024-active-data-collection-operator-learning.pdf) | Theory for when active sampling can beat passive sampling. | Strongest results use simplified/linear operator settings. | Justifies testing active collection, not assuming it always helps. |
| [Winovich et al. — Active operator UQ](04_operator_active_learning/winovich-2025-active-operator-uncertainty.pdf) | Predictive uncertainty for active DeepONet/FNO training. | The uncertainty proxy correlates with actual field error. | Candidate response-aware acquisition method. |
| [Polanska et al. — Physics-based active NO](04_operator_active_learning/polanska-2026-physics-active-neural-operator.pdf) | Use physics residuals to acquire neural-operator training cases. | A computable residual is a useful error proxy. | Applicable only if the shell residual can be evaluated cheaply and consistently. |
| [Satheesh et al. — PICore](04_operator_active_learning/satheesh-2025-picore.pdf) | Physics-informed unlabeled coreset selection. | Physics features rank samples before expensive labels. | Promising pilot alternative, but requires validation against response learning curves. |

FNO and DeepONet foundation papers are archived in the same directory.

## Big idea 5: retain temporal kernels only as a contingency

| Paper | Main idea | Activation condition for this project |
|---|---|---|
| [Király & Oberhauser — Sequential kernels](05_temporal_contingency/kiraly-2019-sequential-kernels.pdf) | Convert a static kernel into a positive-definite kernel on ordered sequences. | Use only after an order-permutation experiment shows constitutive dependence. |
| [Toth & Oberhauser — Signature GP](05_temporal_contingency/toth-2020-signature-gp.pdf) | GP covariance for variable-length paths using signatures. | Use when the GP input truly includes ordered controls/history. |
| [Lemercier et al. — SigGPDE](05_temporal_contingency/lemercier-2021-siggpde.pdf) | Scalable signature-kernel GP computations. | Use if signature GP scaling becomes the bottleneck. |
| [Muca Cirone et al. — Neural signature kernels](05_temporal_contingency/muca-cirone-2023-neural-signature-kernels.pdf) | Learn a signature kernel rather than fixing its path representation. | Requires labeled, genuinely order-dependent response data. |
| [Cuturi — Global alignment kernels](05_temporal_contingency/cuturi-2011-fast-global-alignment-kernels.pdf) | Positive-definite DTW-like alignment kernels. | Consider only when time-warp alignment is physically justified. |

## Recommended program for this project

1. **Specify the state:** initial curvature, current hardening/internal-state
   field, spatial treatment increment, and calibration vector.
2. **Generate a nested pilot:** use LHS for calibration coordinates and a
   nested space-filling/support-point design for spatial fields. Start with
   25–50 simulations.
3. **Learn response coordinates:** compress output-curvature fields with a
   frozen PCA/POD basis fitted only on the pilot training set.
4. **Fit and validate covariance:** compare a functional-input GP, deep kernel,
   and ensemble neural-operator uncertainty. Use held-out likelihood,
   calibration, response-neighbor agreement, and field error—not the
   acquisition objective itself.
5. **Acquire adaptively:** compare random, passive nested design, GP information
   gain, ensemble uncertainty plus diversity, and a physics-residual method.
6. **Activate temporal methods only after evidence:** run matched `AB` versus
   `BA` overlapping-pass experiments with a calibrated history law. If final
   constitutive state/output differs beyond numerical variability, add a
   sequence kernel or recurrent operator; otherwise remain spatial.

## Eight-paper reading order

1. Qian et al. — nested pilot design.
2. Krause et al. — what data-grounded mutual information actually requires.
3. Muehlenstaedt et al. — the simplest functional-input GP baseline.
4. Sung et al. — physics-facing functional-input kernels.
5. Wilson et al., followed immediately by Ober et al. — learned kernels and
   their failure modes.
6. Li et al. — active neural-operator acquisition.
7. Kohl et al. — learning response-aware simulation similarity.
8. Király & Oberhauser only if order dependence is later demonstrated.

## What not to do

- Do not call log-determinant coverage under an arbitrary input RBF kernel
  physical mutual information.
- Do not learn a temporal/path kernel when the simulator has no constitutive
  memory.
- Do not evaluate a selector only by the objective it optimizes.
- Do not learn the response basis, kernel, or normalization using the frozen
  test set.
- Do not cross every calibration point with every spatial field blindly; use a
  balanced/nested product design with repeated anchors for identifiability.
