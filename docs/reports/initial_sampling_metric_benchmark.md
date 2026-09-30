# Effect of spatial treatment kernels on neural-operator sample efficiency

## Current model scope

For the present single-pass, hardening-free simulator, traversal time and order
are not constitutive inputs. The primary initial-design object is therefore the
final spatial top/bottom target-metric field. Ordered-L2, Fréchet, and DTW
designs remain as legacy negative-control baselines; they should not be given a
physical temporal interpretation unless overlap plus an active history law
passes an order-sensitivity test.

## Research question

For a fixed simulation budget, how does the initial spatial treatment design
change the accuracy, robustness, and subsequent active-learning efficiency of a
neural operator for forming?

The experiment separates three objects:

1. **Representation:** final spatial target-metric fields, with action and ordered-path representations retained as baselines.
2. **Similarity:** a positive-definite spatial RBF kernel, action-space Euclidean distance, or a legacy ordered-path distance.
3. **Selection:** random, Latin hypercube, constrained farthest-first maximin, or constrained information gain.

LHS is therefore a baseline rather than a required first stage. It stratifies
coordinates but does not optimize the spatial treatment kernel.

## Initial designs

The generated pilot contains seven designs of 500 samples:

- stratified random;
- Latin hypercube;
- action-coordinate maximin;
- D2 ordered-L2 maximin;
- D2 Fréchet maximin;
- D2 constrained-DTW maximin; and
- spatial target-metric information gain.

All methods have identical scale-class and strip-count totals. Random, maximin,
and spatial-information methods use the same feasible reference ensemble. LHS
is generated independently and is evaluated as a separate space-filling
baseline. A later experiment can replace the discrete reference ensemble with
continuous or on-demand proposal optimization.

The nested evaluation budgets are

\[
N_0\in\{25,50,100,200,500\}.
\]

Use the first \(N_0\) records of each JSONL manifest. Run at least five independent
design seeds before making statistical claims.

## Symmetry and history

For a rectangle with symmetric material and boundary conditions, define

\[
d_{D_2}(T_i,T_j)=\min_{g\in D_2}d(T_i,gT_j),
\qquad
D_2=\{I,F_x,F_y,R_{180}\}.
\]

Only spatial coordinates are reflected. The current spatial-information design
does not contain a traversal-time coordinate. Ordered-path baselines preserve
their original order only as a negative control. If a future calibrated
hardening or plasticity model makes order physically relevant, exclude time
reversal but retain paired order-sensitivity tests. If clamps, material axes, or
robot constraints break a spatial reflection symmetry, remove that
transformation before interpreting results physically.

## Frozen response benchmark

Before training any model, create and version one independent test set. It must
not be used for initial selection, hyperparameter selection, or active-learning
acquisition. Recommended partitions are:

- IID feasible zigzags;
- local, medium, and near-full scale strata;
- reflected copies for equivariance testing;
- paired order reversals only for future history-enabled regimes; and
- boundary and extreme-control challenges.

All sampling methods must use the same training fidelity, solver tolerances,
mesh, output grid, neural-operator architecture, optimizer, epoch budget, and
normalization computed only from their training data.

## Neural-operator endpoints

For a predicted field \(\widehat{u}\) and reference field \(u\), record:

\[
E_{L^2}=\frac{\|\widehat{u}-u\|_2}{\|u\|_2+\epsilon},
\qquad
E_{H^1}=\frac{\|\widehat{u}-u\|_{H^1}}{\|u\|_{H^1}+\epsilon}.
\]

Also record:

- median, mean, 95th-percentile, and worst-case field error;
- displacement, curvature, thinning, and residual-stress quantities of interest;
- error by scale stratum and strip count;
- reflection-equivariance error; and
- reversed-path response discrimination.

Report the full learning curve, not only the error at 500 samples. Useful scalar
summaries are area under the log-budget learning curve and simulations required
to reach a fixed error threshold.

## Does the input distance describe the physics?

After response labels exist, audit each input distance against response distance:

\[
r_{ij}=
\frac{\|\mathcal G(X_i)-\mathcal G(X_j)\|}
     {d(X_i,X_j)+\epsilon}.
\]

Measure Spearman correlation, nearest-neighbor overlap, triplet-order agreement,
and the distribution of \(r_{ij}\). This distinguishes a design that is merely
well spread geometrically from one that is informative about the forming
operator.

## Active-learning phase

After comparing initialization alone, start the same active learner from every
initial design. A generic batch acquisition is

\[
a(X)=U(X)+\lambda D(X,S)-\gamma C(X),
\]

where \(U\) is ensemble epistemic uncertainty, \(D\) is diversity under the
spatial treatment kernel, and \(C\) is fidelity-dependent cost. Freeze acquisition
weights and batch size across initializations. Compare both absolute accuracy
and improvement relative to the accuracy immediately before active learning.

This produces two separate conclusions: which metric gives the best cold-start
design, and whether active learning eventually erases the initialization gap.

## Multi-fidelity use

The manifests should first be screened with the inexpensive simulator. Physical
or high-fidelity labels can then be allocated at the same nested budgets for the
most promising methods. A high-fidelity label must not be silently replaced by
a low-fidelity label in the accuracy comparison; fidelity and initialization
method are separate experimental factors.

## Related literature

The current spatial-sampling framing is organized in
[papers/sampling/README.md](../papers/sampling/README.md), with an
open-access PDF manifest beside it.
