# Overlapping Perpendicular Toolpath Investigation Protocol & Specification

## 1. Authority & Scientific Scope
- **Direct Researcher Authorization**: The researcher explicitly approved switching from the non-overlapping baseline to an overlapping perpendicular toolpath investigation (user directions: *"lets switch to a overlapping perpendicular toolpath"*, *"Center Toolpath B at (0,0)"*, *"10 passes each (20 total)"*, *"n=18 looks good enough"*).
- **Core Scientific Questions**:
  1. *Order effect (changed target metrics)*: In bilayer shell growth with face-shared Voce plastic strain-hardening decay ($\beta = 0.22314355$), does the application sequence of spatially overlapping orthogonal toolpath envelopes ($A \to B$ block, $B \to A$ block, $AB$ alternating, $BA$ alternating) break metric commutativity and produce differing target metric tensors $\bar{A}_{\text{top/bottom}}$ and distinct equilibrium shapes?
  2. *Cadence effect (path-dependent elastic equilibrium)*: With slope feedback disabled (`-sequence_slope_feedback 0`), target metric accumulation is identical across solve modes for a fixed schedule. Does sequential intermediate relaxation (`every_cycle`, 20 discrete elastic equilibria) diverge in final shape from single-stage combined loading (`final_only`, continuation from $\lambda = 0 \to 1$)?
- **Model-Conditional Scope Qualification**: All findings in this investigation represent **model-conditional predictions** under the phenomenological Voce plastic strain-hardening law and Kirchhoff-Love bilayer shell kinematics. Physical identification claims are explicitly deferred and qualified by these constitutive assumptions; no unconditioned physical claims are made. Order contrasts reflect changed-objective effects from non-commutative plastic updates, not branch bifurcations of a shared energy functional.

## 2. Toolpath Geometry & Material Specifications
- **Sheet Geometry**: Rectangular free sheet, $200\text{ mm} \times 300\text{ mm}$ ($u \in [-100, 100]\text{ mm}, v \in [-150, 150]\text{ mm}$), thickness $h = 0.77\text{ mm}$, base mesh resolution parameter `-res .01` yielding nominal vertex spacing $\Delta x = 2 \cdot l_x \cdot \text{res} = 2\text{ mm}$ (30,000 triangular faces, 15,251 vertices).
- **Toolpath A**:
  - Center: $(0, 0)\text{ mm}$, Global Envelope Axis: $0^\circ$ (vertical envelope along $v$).
  - Parameters: $L_v = 140\text{ mm}$, $N = 10$ strips, pitch $= 22.5\text{ mm}$, width $= 8.0\text{ mm}$, interior zigzag tilt $\alpha = 9.13^\circ$.
  - Profile: `center_peak` (top_end_ratio = 0.4, top_profile_power = 2.0).
  - Growth: $g_{\text{top}} = 2.88351 \times 10^{-4}$, $g_{\text{bot}} = 1.45880 \times 10^{-5}$, ortho top = $-0.398916$, ortho bot = $0.960306$.
  - Bounding envelope: $u \in [-94.0, +94.0]\text{ mm}$, $v \in [-74.0, +74.0]\text{ mm}$.
- **Toolpath B** (Centered, $N=18$):
  - Center: $(0, 0)\text{ mm}$, Global Envelope Axis: $90^\circ$ (horizontal envelope along $u$).
  - Parameters: $L_v = 196\text{ mm}$, $N = 18$ strips, pitch $= 9.792\text{ mm}$, width $= 4.0\text{ mm}$, interior zigzag tilt $\alpha = 2.86^\circ$.
  - Profile: `uniform`.
  - Growth: Identical calibrated values to Toolpath A.
  - Bounding envelope: $u \in [-100.0, +100.0]\text{ mm}$, $v \in [-80.33, +80.33]\text{ mm}$.
- **Overlap Domain & Support**:
  - Global envelope overlap: $u \in [-94.0, +94.0]\text{ mm}$, $v \in [-74.0, +74.0]\text{ mm}$ ($188\text{ mm} \times 148\text{ mm}$).
  - Actual co-hit material interaction is evaluated directly on triangular faces receiving plastic hits from both toolpaths (co-hit face support), accounting for finite strip widths (8 mm and 4 mm) and strip tilt angles.
- **Constitutive Hardening Model**:
  $$\text{effective growth} = g_0 \cdot \exp(-\beta \cdot \text{hit\_count}), \quad \beta = 0.2231435513142097, \quad \text{floor} = 0$$
  Hits are face-shared and track cumulative plastic hits across all operations.

## 3. Experimental Matrix
Each toolpath is applied 10 times (20 total applications):
1. **Primary Schedule Arms**:
   - `ab_block`: $A^{10} B^{10}$
   - `ba_block`: $B^{10} A^{10}$
   - `ab_alternate`: $(AB)^{10}$
   - `ba_alternate`: $(BA)^{10}$
2. **Solve Modes**:
   - `final_only`: Continuation from $\lambda=0 \to 1$ in substeps $\Delta \lambda \le 0.1$, solving for the terminal equilibrium of the accumulated 20-cycle growth field.
   - `every_cycle`: Sequential solve yielding 20 successive equilibrium states with intermediate elastic relaxation.
3. **Control Arms**:
   - `a_only`: $A^{10}$ (isolated vertical toolpath baseline)
   - `b_only`: $B^{10}$ (isolated horizontal toolpath baseline)
4. **Preselected Sensitivity Checks** (on `ab_block_final_only`):
   - `ab_block_final_only_repeat`: Identical solver settings, verifying deterministic repeatability.
   - `ab_block_final_only_tight`: Gradient tolerance tightened to $1 \times 10^{-14}$ (vs base $5 \times 10^{-14}$).
   - `ab_block_final_only_fine`: Refined mesh resolution `-res .0075` (nominal spacing $\Delta x = 1.5\text{ mm}$).

## 4. Frozen Alignment & Scoring Protocol
- **Rigid-Body Alignment**: Optimal Procrustes rigid alignment (rotation $R \in SO(3)$ and translation $t \in \mathbb{R}^3$) using singular value decomposition on vertex coordinates to eliminate all 6 unconstrained rigid-body modes.
- **Area-Weighted RMS Deviation**:
  $$d_{\text{RMS}} = \sqrt{\frac{1}{A_{\text{total}}} \sum_{f} A_f \|\bar{x}_f^{(1)} - \bar{x}_f^{(2)}\|^2}$$
- **Maximum Deviation**: $d_{\max} = \max_{i} \|x_i^{(1)} - x_i^{(2)}\|$.
- **Curvature Tensor Extraction**: Principal curvatures $\kappa_1, \kappa_2$ and Gaussian curvature $K = \kappa_1 \kappa_2$ evaluated at sheet centroid and mean over the co-hit face support.
- **Exploratory Effect-Size Screening Gate**:
  - Sheet diagonal $D = \sqrt{200^2 + 300^2} \approx 360.55\text{ mm}$.
  - Effect-size screen: $10^{-4} \times D = 36.1\ \mu\text{m}$.
  - This threshold serves as an exploratory effect-size screen to separate macroscopic shape differences from machine-precision noise. Definite claims regarding discretization bounds or bifurcations are explicitly deferred until both contrasted endpoints are sensitivity-supported.
  - Across solve modes for a fixed schedule (`final_only` vs `every_cycle`), equal target metric tensors and application histories must be verified prior to comparing elastic equilibrium shapes.

## 5. Acceptance Gates & Numerical Stopping Rules
- **Equilibrium Acceptance**:
  - `proof['accepted'] == true`
  - `termination == 'gradient_target'`
  - `curvature == 'positive_pivots'` (strictly positive definite Hessian)
  - `gradient_norm <= gradient_tolerance <= 5e-14` (or `1e-14` for tight variant)
- **Continuation Convergence**:
  - Adaptive step reduction with factor 0.5 upon trial failure; minimum step $\Delta \lambda \ge 0.015625$.
  - Maximum retries per substep: 8.
  - Solver time budget per solve: 1800 s (timeout 72,000 s for sequential 20-cycle; 7,200 s for combined).
- **Stopping Rules**:
  - Allocation overruns, unrecoverable line-search failures, or non-positive-definite Hessians halt execution for human inspection; blind retries are prohibited.

## 6. Inspectable Fixture Evidence Paths
- Combined Continuation (`final_only`):
  - `run/overlapping_sequence/fixtures/ab_block_fixture`
  - `run/overlapping_sequence/fixtures/ba_block_fixture`
- Sequential Intermediate Equilibrium (`every_cycle`):
  - `run/overlapping_sequence/fixtures/ab_block_every_cycle_fixture`
  - `run/overlapping_sequence/fixtures/ba_block_every_cycle_fixture`
- All 4 fixtures completed with `status: accepted`, verified terminal proofs, convergence tables, history rows, and output meshes.
