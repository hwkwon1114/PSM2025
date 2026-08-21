# Mathematical Formulation of Trajectory-Aware Multi-Fidelity Learning

**Predictive representations, neural operators, active experiment design, and world-model learning for history-dependent forming**

**Date:** 20 August 2026

## Abstract

This note formalizes a research program for learning controlled, history-dependent physical
systems from simulations and experiments. The motivating application is English-wheel
forming, but the mathematical problem is more general: a continuous control trajectory acts
on a distributed, partially observed, irreversible system; different numerical models and
physical experiments provide observations at different costs and fidelities; and the learner
must choose which trajectory to evaluate next.

The central proposal separates an information estimator from a trajectory optimizer while
jointly adapting a predictive representation of physical state. The representation is not
assumed to be a small global vector. It may be a latent spatial field learned by a neural
operator. Compression is justified only when the latent state is predictively sufficient for
the admissible future controls. Initial design, representation learning, acquisition, and
multi-fidelity calibration are therefore treated as a coupled sequential problem.

## 1. Controlled history-dependent system

Let the material domain be $\Omega\subset\mathbb{R}^2$. A commanded experiment is a
function-valued control

$$\mathcal{T}(t)=\bigl(\mathbf{x}(t),F_N(t),v(t),r(t)\bigr),\qquad t\in[0,T],$$

where $\mathbf{x}(t)$ is tool position, $F_N(t)$ is normal force, $v(t)$ is speed, and
$r(t)$ encodes tool side or another discrete process mode. Dynamic feasibility defines an
admissible set $\mathcal{U}_{\mathrm{ad}}$ through constraints such as

$$\Vert\dot{\mathbf{x}}\Vert\leq v_{\max},\qquad \Vert\ddot{\mathbf{x}}\Vert\leq a_{\max},\qquad 0\leq F_N\leq F_{\max}.$$

Let $s_k$ denote the complete physical state after pass $k$. In forming, $s_k$ can include
geometry, elastic strain, plastic strain, hardening variables, thickness, and residual stress.
The true evolution is

$$s_{k+1}=G_R(s_k,z_k,\boldsymbol\theta)+w_k,$$

where $G_R$ is the unknown real transition operator, $\boldsymbol\theta$ contains material
and process parameters, and $w_k$ represents unresolved variability. The realized actuation
$z_k$ need not equal the command $\mathcal{T}_k$:

$$z_k\sim p_\eta\bigl(z_k\mid \mathcal{T}_k,s_k\bigr).$$

This distinction prevents contact, slip, force-control, and machine errors from being
incorrectly absorbed into a material hardening law.

The experiment returns an incomplete noisy observation

$$y_k=H(s_k)+\epsilon_k,$$

where $H$ can represent profilometry, DIC, force sensors, thickness measurements, or a
simulation output map. The available history is

$$h_k=(y_0,\mathcal{T}_0,y_1,\ldots,\mathcal{T}_{k-1},y_k).$$

The system is generally non-Markovian in $y_k$, even when it is Markovian in the inaccessible
complete state $s_k$.

## 2. Path space, measures, and initial sampling

There is no canonical uniform probability measure on an infinite-dimensional function
space. An initial design must specify three objects:

1. an admissible path space $\mathcal{X}_{\mathrm{ad}}$;
2. a reference probability measure $\mu$ on that space; and
3. a metric or kernel defining meaningful path separation.

A practical finite-bandwidth representation is

$$\mathcal{T}(t)=\sum_{j=1}^{m}c_jB_j(t),$$

where $B_j$ are spline, Fourier, wavelet, or Karhunen--Lo\`eve basis functions. The
coefficient dimension $m$ may vary between paths. Complexity is controlled by physical
bandwidth, curvature, contact-footprint, and actuator constraints rather than by prescribing
a zigzag topology.

If a decoder $g:\mathcal{Z}\rightarrow\mathcal{X}_{\mathrm{ad}}$ maps reduced coordinates
$q$ to a path, uniform sampling in $q$ is not uniform on the decoded manifold. The induced
local volume element is

$$dV_g(q)=\sqrt{\det\bigl(J_g(q)^\top J_g(q)\bigr)}\,dq.$$

Consequently, decoder distortion must be measured or compensated when uniform manifold
coverage is desired.

### 2.1 Canonical path metric

After canonical arc-length or physical-time parameterization, define

$$d_X^2(\mathcal{T}_1,\mathcal{T}_2)=d_{\mathrm{geom}}^2+\alpha_Fd_F^2+\alpha_vd_v^2+\alpha_hd_h^2.$$

One useful geometric term is an $H^1$-type metric

$$d_{\mathrm{geom}}^2=\int_0^1\!\Vert\mathbf{x}_1-\mathbf{x}_2\Vert^2ds+\rho\int_0^1\!\Vert\mathbf{x}'_1-\mathbf{x}'_2\Vert^2ds.$$

The history term $d_h$ distinguishes visit count, traversal direction, and process state.
Dynamic time warping should not be used without care because it can erase timing and order
that are physically meaningful.

### 2.2 Descriptors are auxiliary coordinates

Let $\phi(\mathcal{T})\in\mathbb{R}^p$ contain interpretable descriptors such as length,
curvature, overlap, swept area, boundary distance, dose, and number of reversals. Coverage in
$\phi$-space transfers to path-space coverage only under an inverse-Lipschitz condition. A
full geometry-preserving guarantee would require

$$m\,d_X(\mathcal{T}_1,\mathcal{T}_2)\leq\Vert\phi(\mathcal{T}_1)-\phi(\mathcal{T}_2)\Vert\leq M\,d_X(\mathcal{T}_1,\mathcal{T}_2).$$

Finite handcrafted descriptors do not satisfy this globally. They are therefore used for
stratification and protected quotas, not as proof of representation diversity.

### 2.3 Kernel and distribution coverage

Let $k_X$ be a characteristic kernel on canonical path representations. Its reproducing
kernel Hilbert space distance defines maximum mean discrepancy

$$\mathrm{MMD}_{k_X}(\widehat{\mu}_N,\mu)=\left\Vert\frac{1}{N}\sum_{i=1}^Nk_X(\mathcal{T}_i,\cdot)-\int k_X(\mathcal{T},\cdot)d\mu(\mathcal{T})\right\Vert_{\mathcal{H}}.$$

An initial design can combine low MMD, maximin path distance, kernel log determinant, and
descriptor quotas. For a kernel Gram matrix $K_S$ of a selected batch $S$, a D-optimal
diversity criterion is

$$S^*=\arg\max_{|S|=B}\log\det\bigl(K_S+\varepsilon I\bigr).$$

The design is evaluated in the actual ordered input representation, not only in a PCA plot
of descriptors.

## 3. When can a high-dimensional state be compressed?

There is no lossless continuous encoding of an arbitrary open subset of
$\mathbb{R}^D$ into $\mathbb{R}^d$ when $d<D$. Compression requires additional structure:
a low-dimensional attractor, finite system order, a low-dimensional manifold, or a task that
does not require all information in the original state.

For forming, a small global vector may be particularly implausible because residual stress,
plastic strain, and hardening are spatial fields. A safer representation is a latent field

$$z_k(\mathbf{x})=E_\theta[h_k](\mathbf{x}),\qquad \mathbf{x}\in\Omega,$$

or a hierarchical state

$$z_k=\bigl(z_k^{\mathrm{global}},z_k^{\mathrm{local}}(\mathbf{x})\bigr).$$

The latent representation reduces channels, resolution, or irrelevant detail without
assuming that all distributed memory collapses into a few scalars.

### 3.1 Predictive sufficiency

The correct equivalence relation is determined by future consequences. An encoding is
predictively sufficient if, for every admissible future action sequence $U$,

$$p\bigl(Y_{k+1:k+H}\mid h_k,U\bigr)=p\bigl(Y_{k+1:k+H}\mid z_k,U\bigr).$$

Equivalently, the conditional mutual information vanishes:

$$I\bigl(Y_{k+1:k+H};h_k\mid z_k,U\bigr)=0.$$

This is the desired guarantee. It is rarely verifiable exactly for nonlinear physical
systems, so it becomes an empirical hypothesis tested across future controls, horizons,
path families, and fidelities.

### 3.2 Information bottleneck interpretation

The predictive information-bottleneck objective seeks a compact code that preserves future
information:

$$\min_E\ I(z_k;h_k)-\beta I\bigl(z_k;Y_{k+1:k+H}\mid U\bigr).$$

The first term rewards compression; the second rewards prediction. This formalizes the
tradeoff but does not establish that a chosen finite dimension is sufficient. The answer
depends on the data distribution and the action sequences included during training.

## 4. Neural-operator predictive world model

A neural operator is retained as the map between spatial function spaces. A JEPA-style
objective changes how its representation is trained; it does not replace the operator with
a conventional autoencoder.

Let the state encoder, action encoder, predictor, and target encoder be

$$z_k=E_\theta(h_k),\qquad a_k=A_\psi(\mathcal{T}_k),$$

$$\widehat{z}_{k+1}=P_{\omega,\ell}(z_k,a_k),\qquad \overline{z}_{k+1}=E_{\bar{\theta}}(h_{k+1}).$$

The fidelity index $\ell$ selects a coarse model, refined solver, or real experiment. The
basic latent prediction loss is

$$\mathcal{L}_{\mathrm{JEPA}}=\Vert\widehat{z}_{k+1}-\mathrm{sg}(\overline{z}_{k+1})\Vert^2,$$

where $\mathrm{sg}$ denotes stop-gradient and the target parameters $\bar{\theta}$ evolve
slowly. For a latent field, the norm integrates over $\Omega$.

A physical output head preserves the original full-field task:

$$\widehat{s}_{k+1}=D_{\mathrm{field}}(\widehat{z}_{k+1}).$$

The combined objective is

$$\mathcal{L}=\mathcal{L}_{\mathrm{JEPA}}+\lambda_y\mathcal{L}_{\mathrm{field}}+\lambda_p\mathcal{L}_{\mathrm{physics}}+\lambda_c\mathcal{L}_{\mathrm{collapse}}.$$

Here $\mathcal{L}_{\mathrm{field}}$ measures geometry, curvature, thinning, or stress
prediction; $\mathcal{L}_{\mathrm{physics}}$ enforces known invariants or residuals; and
$\mathcal{L}_{\mathrm{collapse}}$ preserves latent variance and covariance.

### 4.1 Multi-horizon loss

Slow hardening variables may not influence the next pass strongly. Multi-horizon training
forces the representation to preserve long-lived memory:

$$\mathcal{L}_{H}=\sum_{j=1}^{H}w_j\Vert\widehat{z}_{k+j}-\mathrm{sg}(\overline{z}_{k+j})\Vert^2.$$

The predictor is action-conditioned at every step, so the learned state is tested under
different possible future controls rather than only under the behavior policy that generated
the current data.

## 5. Representation failure and diagnostics

### 5.1 Conditional sufficiency probe

After learning $z=E(h)$, compare a probe using only latent state and future control,

$$\widehat{Y}=Q_0(z,U),$$

with a residual probe that also receives the original history,

$$\widehat{Y}=Q_1(z,U,h).$$

If $Q_1$ improves held-out prediction materially, then $z$ is not predictively sufficient.

### 5.2 Latent collision search

Seek histories that collide in latent space but diverge under some future control:

$$\Vert E(h_i)-E(h_j)\Vert\leq\varepsilon,$$

$$\max_{U\in\mathcal{U}_{\mathrm{ad}}}D\bigl(p(Y\mid h_i,U),p(Y\mid h_j,U)\bigr)\gg0.$$

The maximizing $U$ is an active experiment specifically designed to falsify the current
representation.

### 5.3 Dimension and bottleneck sweep

No single latent dimension is assumed. Compare latent vector dimensions, latent field
resolutions, and channel counts using one-step error, long-horizon rollout error, branch
classification, uncertainty calibration, and held-out path-family error. A performance
plateau is evidence only for the tested distribution, not a universal proof.

### 5.4 Representation drift

Because acquisition changes the data distribution, the encoder changes after each round.
Store raw histories rather than only latent codes, re-encode the replay buffer, and constrain
drift on a fixed anchor set $\mathcal{D}_A$:

$$\mathcal{L}_{\mathrm{anchor}}=\sum_{h\in\mathcal{D}_A}\Vert E_{n+1}(h)-R_nE_n(h)\Vert^2,$$

where $R_n$ aligns successive latent coordinate systems. Acquisition is computed using a
frozen model snapshot within each round.

## 6. Bayesian information acquisition

Let $b_k=p(s_k,\boldsymbol\theta,\delta\mid\mathcal{D}_k)$ be the current belief over
state, calibration parameters, and model discrepancy. For a candidate design $d$, expected
information gain is

$$\mathrm{EIG}(d)=\mathbb{E}_{y\mid d}\left[\mathrm{KL}\bigl(p(\boldsymbol\theta\mid y,d)\Vert p(\boldsymbol\theta)\bigr)\right].$$

This equals conditional mutual information:

$$\mathrm{EIG}(d)=I(\boldsymbol\theta;y\mid d)=H(\boldsymbol\theta)-\mathbb{E}_{y\mid d}H(\boldsymbol\theta\mid y,d).$$

The Fisher information matrix is a local alternative:

$$F(\boldsymbol\theta;d)=\mathbb{E}_{y\mid\boldsymbol\theta,d}\left[\nabla_{\boldsymbol\theta}\log p(y\mid\boldsymbol\theta,d)\nabla_{\boldsymbol\theta}\log p(y\mid\boldsymbol\theta,d)^\top\right].$$

D-optimality maximizes $\log\det F$, A-optimality minimizes
$\mathrm{tr}(F^{-1})$, and E-optimality maximizes the smallest eigenvalue. Under local
Gaussian assumptions, D-optimal Fisher design approximates entropy reduction.

Parameter information is not identical to predictive information. For operator learning,
define a target distribution $\nu$ over future histories and controls and a predictive risk

$$R_R(M)=\mathbb{E}_{(h,U)\sim\nu}\mathbb{E}_{Y\sim G_R}\bigl[L(M(h,U),Y)\bigr].$$

An acquisition can target expected reduction $\mathbb{E}[\Delta R_R]$ rather than only
parameter entropy.

### 6.1 Latent epistemic uncertainty

Use an ensemble of predictors

$$\widehat{z}_{k+1}^{(m)}=P_{\omega_m,\ell}(z_k,a_k).$$

Predictor disagreement is

$$U_{\mathrm{latent}}(d)=\mathrm{tr}\,\mathrm{Cov}_m\bigl[\widehat{z}_{k+1}^{(m)}\bigr].$$

This is an approximation to epistemic uncertainty, not a guarantee. It must be calibrated
on held-out trajectories and separated from irreducible experimental variability.

### 6.2 Composite cost-aware acquisition

For trajectory $\mathcal{T}$ and fidelity $\ell$, define

$$\mathcal{A}(\mathcal{T},\ell)=\frac{U_{\mathrm{pred}}+\lambda_\theta U_\theta+\lambda_\delta U_\delta+\lambda_oU_{\mathrm{order}}+\lambda_rU_{\mathrm{raw}}}{c_\ell(\mathcal{T})}-\lambda_sR_{\mathrm{safety}}.$$

The terms measure predictive uncertainty, parameter information, discrepancy reduction,
order-effect information, raw path-space exploration, and risk. The raw-space term prevents
a self-confirming latent model from ignoring distinctions absent from its current encoding.

## 7. Ergodic trajectory synthesis

The information estimator determines where or under what state-action conditions a
measurement would be valuable. Ergodic control is an optional inner optimizer that converts
an information density into a dynamically feasible continuous path.

Let $\Phi(q)$ be an information density on an augmented state-action coordinate $q$, not
only physical position. Expand it in basis functions $F_j$:

$$\phi_j=\int\Phi(q)F_j(q)dq.$$

For a trajectory $q(t)$, its time-average coefficients are

$$c_j=\frac{1}{T}\int_0^TF_j(q(t))dt.$$

The ergodic mismatch is

$$\mathcal{E}[q]=\sum_j\Lambda_j(c_j-\phi_j)^2.$$

An inner planner minimizes

$$\mathcal{J}_{\mathrm{path}}=\mathcal{E}[q]+\lambda_u\int_0^T\Vert u(t)\Vert^2dt+\lambda_sR_{\mathrm{safety}}.$$

Because forming changes the sheet irreversibly, $q$ must include history, visit count,
hardening belief, direction, force, and side. Purely spatial ergodicity would incorrectly
treat repeated visits as interchangeable.

## 8. Multi-fidelity calibration

Let $G_0$ be the cheapest operator and define successive discrepancies

$$G_\ell=G_0+\sum_{j=1}^{\ell}\Delta_j.$$

For reality,

$$G_R=G_{HF}(\cdot;\boldsymbol\theta_{cal})+\Delta_R+\epsilon_R.$$

The calibration parameters $\boldsymbol\theta_{cal}$ and discrepancy $\Delta_R$ must be
modeled separately to prevent missing physics from being absorbed into implausible material
parameters.

A shared latent operator can use fidelity-specific predictors

$$P_\ell(z,a)=P_{\mathrm{shared}}(z,a)+\Delta P_\ell(z,a).$$

The next query chooses both trajectory and fidelity:

$$\bigl(\mathcal{T}^*,\ell^*\bigr)=\arg\max_{\mathcal{T},\ell}\mathcal{A}(\mathcal{T},\ell).$$

Resolution-only fidelities are simpler than changes in governing physics. Validation must
therefore separate mesh refinement, model-form discrepancy, and simulation-to-real transfer.

## 9. Closed-loop active world-model algorithm

At acquisition round $n$:

1. Maintain raw replay data $\mathcal{D}_n$ and belief $b_n$.
2. Train or update the state encoder, action encoder, predictor ensemble, target encoder,
   physical head, and discrepancy model.
3. Re-encode all raw replay data and audit latent drift, effective rank, and collisions.
4. Generate feasible candidates from human primitives, free-form paths, protected reversal
   pairs, and adaptive-complexity proposals.
5. Score candidate trajectory--fidelity pairs using $\mathcal{A}$.
6. Optimize the path continuously, optionally using an ergodic inner planner.
7. Validate feasibility and uncertainty using a frozen model snapshot.
8. Execute the selected simulation or experiment.
9. Record commanded and realized actuation plus intermediate observations.
10. Update $\mathcal{D}_{n+1}$ and repeat.

The representation and dataset co-evolve:

$$\mathcal{D}_n\rightarrow(E_n,P_n)\rightarrow\mathcal{A}_n\rightarrow(\mathcal{T}_{n+1},\ell_{n+1})\rightarrow\mathcal{D}_{n+1}.$$

This feedback is the central research object. A poor initial design can hide future-relevant
directions, while an unstable representation can invalidate acquisition geometry.

## 10. Validation hypotheses

### H1: Predictive sufficiency

Adding raw history to a strong latent-state probe should not materially improve future
prediction on held-out action families.

### H2: Representation-aware initial design

A hybrid design using canonical path kernels, descriptor quotas, and collision diagnostics
should reduce held-out operator error more efficiently than zigzag-only, primitive-only,
descriptor-only, or random designs.

### H3: Active state discovery

Experiments optimized to distinguish latent collisions should reveal history variables and
order effects more efficiently than generic uncertainty sampling.

### H4: Multi-fidelity allocation

Cost-aware trajectory--fidelity acquisition should achieve lower real-fidelity predictive
risk at equal total cost than fixed fidelity ratios or HF-only sampling.

### H5: Free-form path discovery

After independent-oracle validation, paths outside the human primitive library should provide
either higher information gain or improved downstream forming performance at equal risk and
cost.

## 11. Evaluation metrics and falsification tests

All learning curves use cumulative measured oracle cost on the horizontal axis. Report:

- one-step and multi-step full-field error;
- worst-case error over held-out path families;
- latent collision rate and conditional-sufficiency probe improvement;
- latent effective rank, anchor drift, and representation covering radius;
- MMD between acquired and target path measures;
- interval coverage, negative log likelihood, and epistemic calibration;
- order-effect detection relative to repeatability noise;
- posterior contraction and bias of calibration parameters;
- LF--HF--real discrepancy prediction;
- safety violations and failed experiments;
- downstream trajectory-optimization regret.

The representation hypothesis is falsified if raw history consistently improves prediction,
latent collisions cannot be resolved by increasing capacity or active experiments, or an
uncompressed neural operator dominates at comparable cost. The active-sampling hypothesis is
falsified if information scores do not predict realized error reduction on controlled
benchmarks.

## 12. Recommended experimental sequence

1. Analytic multi-fidelity functions validate acquisition and cost accounting.
2. J2 plasticity or Bouc--Wen hysteresis validates memory, reversal, latent sufficiency, and
   active collision tests.
3. AL4PDE validates active operator learning against established baselines.
4. PDEControlGym validates continuous constrained action trajectories.
5. Multi-resolution operator benchmarks validate field-valued fidelity selection.
6. Paired simulation and experiment datasets validate calibration and discrepancy.
7. The English-wheel hierarchy validates free-form paths, order, and real forming utility.

The first implementation should not begin with a large encoder. It should compare a direct
uncompressed operator, handcrafted state, autoencoder state, predictive state, and
JEPA-regularized latent-field operator on a controlled history-dependent benchmark.

## 13. Research gap

Reduced-dimensional sampling, functional Bayesian optimization, neural operators, predictive
state representations, JEPA objectives, ergodic exploration, and multi-fidelity calibration
all have established literatures. The proposed gap is their coupling:

> Jointly learn and actively falsify a predictive latent field for a controlled,
> history-dependent physical operator while selecting continuous trajectories and fidelity
> levels under cost, safety, representation drift, and simulation-to-reality discrepancy.

The novelty is not compression by itself. It is the auditable co-evolution of representation,
experiment design, trajectory synthesis, and multi-fidelity physical knowledge.

## References

1. Tishby, Pereira, and Bialek, The Information Bottleneck Method,
   https://arxiv.org/abs/physics/0004057.
2. Littman, Sutton, and Singh, Predictive Representations of State,
   https://papers.nips.cc/paper_files/paper/2001/hash/1e4d36177d71bbb3558e43af9577d70e-Abstract.html.
3. Assran et al., Self-Supervised Learning from Images with a Joint-Embedding Predictive
   Architecture, https://openaccess.thecvf.com/content/CVPR2023/.
4. Bardes et al., V-JEPA: Latent Video Prediction for Visual Representation Learning,
   https://openreview.net/forum?id=WFYbBOEOtv.
5. Lu et al., Learning Nonlinear Operators via DeepONet,
   https://doi.org/10.1038/s42256-021-00302-5.
6. Li et al., Fourier Neural Operator for Parametric Partial Differential Equations,
   https://openreview.net/forum?id=c8P9NQVtmnO.
7. Kennedy and O'Hagan, Bayesian Calibration of Computer Models,
   https://doi.org/10.1111/1467-9868.00294.
8. Huan and Marzouk, Simulation-Based Optimal Bayesian Experimental Design,
   https://doi.org/10.1016/j.jcp.2012.08.013.
9. Miller et al., Ergodic Exploration of Distributed Information,
   https://arxiv.org/abs/1708.09352.
10. Simon-Gabriel and Sch\"olkopf, Kernel Distribution Embeddings,
    https://jmlr.org/papers/v19/16-291.html.
11. Musekamp et al., Active Learning for Neural PDE Solvers,
    https://openreview.net/forum?id=x4ZmQaumRg.
12. Li et al., Multi-Resolution Active Learning of Fourier Neural Operators,
    https://proceedings.mlr.press/v238/li24k.html.
