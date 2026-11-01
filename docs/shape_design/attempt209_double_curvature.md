# Attempt 209 double-curvature toolpath family (exploratory design)

## Status
**Current state (2026-10-31):** all nine turned-over 2 mm cases and both 1 mm endpoint solves have accepted result records. The six-case continuation was independently reviewed in `turned_over_continuation_review.md` before execution. Saved nine-case PCA and endpoint coarse/fine comparison are reported below; they do not establish mesh convergence or physical validity. No further simulations are authorized by this status update.

**Nine-case and mesh analysis (solver-free Slurm 6148109).** `shape_design/analyze_mesh_check.py` saved input hashes, 2 mm fits, 1 mm endpoint fits and PCA to `run/shape_design/mesh_comparison/results.json` and `pca.npz`. The nine 2 mm turned-over height fields give PC1 99.3909% explained variance and descriptive correlation 0.9634 with fitted Gaussian curvature. PC1 is sign-oriented toward positive correlation with k and is not strictly monotone: the score decreases slightly from k=+3 to k=+4. Coarse fits are saddle-like for k=−4 and dome-like for k=−3 through +4 using the prespecified quadratic curvature-ratio rule; this classification alone does not establish globally uniform curvature.

Both 1 mm endpoint outputs are accepted with gradient-target termination, positive pivots and zero mismatches on 54,560 faces. They retain the 2 mm quadratic classifications and PC1 endpoint ordering when projected onto the fixed 2 mm PCA basis. Height differences at exactly nested coarse vertices after separate full-grid rigid alignment and mean removal are RMS 0.00813/0.00455 mm, unweighted absolute p95 0.01516/0.00930 mm, and maximum 0.02803/0.01869 mm for k=−4/+4. For k=−4 the 15 mm local-fit negative-Gaussian-curvature treated-area fraction is 40.64%/40.31% at 2/1 mm; the quadratic saddle ratio is near the 0.2 threshold (0.207/0.216). These two meshes and two endpoints do not prove convergence; sampling, alignment and model limitations apply. The one-face pilot failure and extrapolation beyond the 0.77 mm calibration are unchanged.

`shape_design/plot_mesh_comparison.py` produces `run/shape_design/mesh_comparison_figures/family_pca_mesh_comparison.pdf` and `.png`, with caption and SHA-256 provenance; it verifies saved input hashes. The PNG and rendered PDF page were inspected (Slurm 6184895 and 6184940); journal dimensions and accessibility simulations were not checked. The direct sign, scale and area fractions are descriptive simulation outcomes, not physical validation.

The original six-pilot gate failed. The researcher explicitly revised the scope to turned-over-only continuation; the one-face failure remains part of the record. The chronology below records the state at each earlier stage.

**Pilot outcome (Slurm array 6034575; tracked job 10301810-zfnc): STOPPED at prescribed gate.** Five of six cases passed the solver acceptance, native-state/VTP/grid correspondence and exact 13,640-face hit-count checks; `oneface k=-4` failed after 2365 s with solver exit -6, `zigzag_sequence: equilibrium corrector failed at cycle 32, lambda=0.921875, gradient_norm=0.000000, termination=indefinite_curvature.` This is a numerical failure, not proof that a shape is impossible. At that stopping point no remaining sweep or 1 mm runs had launched; no retry or tolerance change occurred. Slurm 6040811 computed the predeclared quadratic descriptor on the five saved accepted shapes: `flip k=-4` saddle-like, K_fit −2.5209906e−8 mm⁻² and curvature ratio 0.2069 (near the prespecified 0.2 classification boundary); `flip k=0` dome-like, K_fit 2.9319634e−7, ratio 0.8965; `flip k=+4` dome-like, K_fit 3.4235715e−7, ratio 0.8304. `oneface k=0/+4` are cylinder-like (ratios 0.0493/0.0791). These are coarse-mesh quadratic-fit labels, not full local-curvature or PCA analysis, verified mesh convergence, or physical validation. The observed transition is promising but not proof of continuity. The failed pilot blocked the original predeclared sweep. The researcher subsequently chose the revised turned-over-only scope below; this does not erase or reclassify the failure.

**Final independent review:** `docs/shape_design/preexecution_final_review.md` returned APPROVED_WITH_CAVEATS for plan SHA-256 `0bb8a542cd8f1656a193aaa7dd1ed5e9d458644a37857694f7d59f014ce1e311`. Immediately before pilot, the lead recomputed that hash and all 13 plan input hashes with no mismatch; the pilot output directory did not exist. Manual pilot-to-sweep gate and the explicit 3 h scheduler limit for the 1 mm checks remain mandatory. Fine-mesh projection and full reporting are not yet implemented or reviewed. No solver results or physical validation are implied by the approval.

**Treatment proxy qualification:** the Frobenius norm is taken after averaging tensors, so cancellation can hide local imbalance; the denominator comes from a distinct oneface dose and geometry. Zero-reference tiles are excluded: 46 tiles at k=−4, 42 at k=+4 and 50 otherwise; the reviewer checked all omitted tiles have zero flip hits. Tile maxima differ substantially at 1 mm versus 2 mm, and this diagnostic establishes neither curvature nor mesh convergence.

**First pre-execution review (`verifier`, Astra): BLOCKED** — `docs/shape_design/preexecution_review.md`. Three blocking issues were repaired as follows:
1. **Curl claim.** The 7% ordering figure assumed one hit per pass on every face. It is withdrawn and replaced by the geometry-based diagnostic below.
2. **Float32 coordinates.** Analysis and hit checks now use the full-precision solver state, cross-checked against the VTP meshes and the Python copy of the solver grid. The region tolerance is 1e-3 mm.
3. **1 mm time envelope.** The 1 mm check uses Slurm `-t 03:00:00` for its 7200 s limit. A handled SIGTERM attempts to kill the solver group and record `result.json`; abrupt termination cannot guarantee a record.

Non-blocking fixes:
- The mirror test now compares strips by index and rejects an unmirrored pass.
- `run_case.py` refuses to run unless code, binary, source, timeout and generated-sequence identities match the reviewed plan.
- The pilot-to-sweep gate is enforced by the lead, not code.

**Preparation (no solver).** Code: `shape_design/family.py` (toolpaths, solver-grid copy, hardened-treatment emulator), `run_case.py` (one solve via the calibration runner, acceptance and correspondence checks), `analysis.py`, `run_family.sbatch`, `dry_run.py`, `tests/test_family.py`.
- Solver-free tests: 37 passed, with `RuntimeWarning` treated as an error (tracked job `10301752-ig28`); the same job regenerated the plan and curl diagnostic. No solver was run.
- `dry_run.py` generated all 18 family sequences and both 1 mm checks. Compared with the Attempt 209 Run 1 command, only `-lx`, `-ly`, `-res`, `-h_total` (and `-native_hybrid_seconds` for 1 mm) differ.
- Plan: `run/shape_design/plan/plan.json`, SHA-256 `0bb8a542cd8f1656a193aaa7dd1ed5e9d458644a37857694f7d59f014ce1e311`. It records source, analysis, binary, launcher, test and sequence identities, and timeouts.
- **Nearly constant work:** each region receives `4 ± k` rounds, so its two regions total 8 rounds. Total active strip-passes are 1120, 1104 and 1088 for `k = −4, 0, +4`. The number of partial-mask passes varies from 16 (`k = 0`) to 32 (`k = ±4`).

**Residual-curl diagnostic (pre-solve; `run/shape_design/plan/curl_diagnostic.json`, SHA-256 `578c9c3c32c1550dc06ce0f5cfd642fcb58b391b94e01036a670927a2a12fde3`).** `family.treatment` emulates every hit in solver order with the actual strip geometry, shared hardening and layer tensors, linearized to additive small strains. For the `flip` variant, it compares the mean top-minus-bottom layer difference with the same-`k` `oneface` value:

| Measure | Range across `k` (2 mm) |
|---|---|
| Whole treated rectangle | 1.7–4.8% |
| 20 × 20.8 mm tiles, median | 2.6–5.7% |
| 20 × 20.8 mm tiles, maximum | 8.4% (`k = +1`) to 27% (`k = +4`) |

The 1 mm values are similar. These diagnose treatment, not equilibrium shape: local one-sided bands remain where front and mirrored back footprints differ, especially for crosswise passes and at region boundaries. The earlier face-level ratio and its claim of 0% one-sided faces at `k=0` were invalid and are withdrawn: their denominator used the distinct oneface footprint. The independent re-review counted 1,852 exclusively one-face triangles among 10,365 flip-hit triangles within the treated rectangle at 2 mm and `k=0` (about 17.9%); this counts hit coverage, not residual curvature. See `docs/shape_design/preexecution_rereview.md`.

**Continuation evidence.** Array `6066857` completed the six missing turned-over cases, all accepted in 365–1242 s with zero hit mismatches. Array `6070182` completed the two 1 mm cases: `k=-4` in 2137.74 s and `k=+4` in 5569.78 s. Both records report gradient-target termination, positive pivots and zero mismatches on 54,560 faces. These timings and acceptance fields were inspected during cleanup; the later coarse/fine analysis is recorded in the current-state section. Saved collaborator workbooks show the explicitly labeled 2 mm pilots, not refined results. Existing delivered exports and review snapshots are retained unchanged.

## Researcher request and decisions
- 2026-10-30: using the frozen Attempt 209 parameters, design an interesting doubly curved shape (dome or saddle) on a **124 mm (feed/length) × 220 mm (width) × 0.61 mm** sheet. Magnitudes need not match; the shapes should show an interpretable trend, e.g. under PCA. Method left to the agent.
- The researcher approved the staged plan, allowed the sheet to be **rotated 90°** for crosswise passes, and allowed it to be **turned over** so the wheel works the opposite face.

## Question, role and interpretation
Within the calibrated model, can one feasible toolpath parameter move the predicted shape continuously from saddle to dome?

This is **model-based exploratory design**, not validated prediction:
- Attempt 209 was calibrated on Run 1 only, at 0.77 mm thickness and one-face feed-direction passes. Thinner sheets, crosswise passes and the representation of a back-face pass are extrapolations.
- Trends are evidence about the model's response. A null or one-sided result—for example, persistent cylindrical curl—is also reportable.

## Fixed model
- Parameters copied, not refitted, from `run/convention_corrected/run6/evaluations/attempt_0001/case_00/sequence.json`: `gtop=2.8835116227798787e-4`, `gbot=1.4587994001079146e-5`, `ortho_top=-0.3989157893255133`, `ortho_bottom=0.960306323937336`, Voce hit-count hardening `beta=0.2231435513142097`, `floor=0`, `face_shared`, `plastic_hits`, uniform profile.
- Solver arguments copied from `run/convention_corrected/run6/proposed_config.json` (`final_only`, native hybrid, 5e-14 gradient gate, positive-pivot acceptance, 3600 s), except geometry: `-lx 0.11 -ly 0.062`, `-h_total 0.00061`, `-res 0.002/0.22`, giving a 2 mm right-angle mesh (111 × 63 vertices). Public coordinates are x = width (u), z = length/feed (v + 65 mm), y = height. The historical CSV schema instead uses y = v + 62 mm for feed and z = 0 for the nominal material plane; it is not the public figure frame.
- No clamps; one equilibrium after all passes, as in the calibration.

## Toolpath family
All passes use the Run 1 strip pitch `130 tan(1°) = 2.269 mm` and the 2 mm treated band width, with negative tilt like Run 1.
- **Feed pass F** (rotation 0°): strokes along the 124 mm direction, `lv = 104 mm`, 90 strips (88 inclined), covering about 200 mm across the width.
- **Cross pass C** (sheet rotated, 90°): strokes along the 220 mm direction, `lv = 200 mm`, 48 strips, covering about 104 mm along the length.
- Both cover the same central ~200 × 104 mm rectangle, leaving ~9 mm untreated margins.

**Regions.** A strip is *central* if its stepping-axis position is within the middle half of its pattern (`|ξ| ≤ 0.5`); otherwise *edge*. Inactive strips receive zero growth and do not count as hits.

**Family parameter.** `k ∈ {-4,…,4}`: central strips are worked in `4 + k` rounds and edge strips in `4 − k` rounds. `k = +4` works only the center (expected dome-like), `k = −4` only the edges (expected saddle-like), `k = 0` uniformly. Each round contains four wheel passes, ordered by the variant.

**Variants** (same pass count and positions):
- `flip`: `[F front, F back, C back, C front]`, with the face for round `j` assigned by the Thue–Morse sequence. This order was chosen in advance under an idealized one-hit-per-pass hardening model, not simulations. Its actual geometry-based residual curl is reported in the diagnostic above.
- `oneface`: `[F, F, C, C]`, all on the front face.

**Back-face pass representation (assumption).** Turning the sheet over about its v axis mirrors `u → −u`. In material coordinates, a pattern `(θ, α)` therefore becomes `(−θ, −α)`. The calibrated worked/unworked layer parameters swap: `gtop ↔ gbot`, `ortho_top ↔ ortho_bottom`. Hit history remains shared by the face.

## Known-answer and acceptance checks
Before solving, with no solver:
1. The Python strip builder reproduces the C++ `buildZigZagStrips` formula and footprint checks.
2. Masks are symmetric. Each region has the intended number of active rounds, and every sequence has the documented pass count.
3. A mirrored back pass equals the mirror image of the front pass, strip by strip. The F set is mirror-symmetric, so mirroring changes only its traversal order.
6. The Python grid copy matches C++ vertex and face counts. The hardened-treatment emulator gives the exact single-hit tensor, the repeated hit scaled by `1 + e^{-β}`, and front-plus-back cancellation to `1 − e^{-β}`.
4. The JSON schema and frozen parameters/hardening match the source exactly.
5. Analysis fixtures: analytic dome, saddle and cylinder give the correct quadratic-fit curvature signs. Height extraction is invariant to rigid motion, and PCA recovers a known one-parameter family.

After each solve:
- Use the calibration runner's `check_output` acceptance.
- Initial mesh extent is 220 × 124 mm.
- The full-precision solver state matches both VTP meshes within 1e-4 mm, its faces match, and its reference vertices match the Python grid within 1e-9 mm.
- The solver's per-face `total_hit_count` equals the Python prediction from those coordinates.

## Execution and stopping
1. **Review:** independent Astra review of protocol, code and identities.
2. **Pilot:** `k ∈ {−4, 0, +4}` for both variants, 6 solves.
3. **Revised sweep (researcher decision after pilot):** the three `flip` pilots were accepted within 3600 s with all checks. After independent review of this scope, run only the six missing `flip` values `k = −3, −2, −1, +1, +2, +3`, once each, with the frozen plan, solver and acceptance criteria. Keep all original pilot results and the `oneface k=-4` failure visible. If any new case fails, times out or lacks a valid record, stop after the submitted work finishes and report; do not retry or relax tolerances. Do not launch any additional `oneface` cases.
4. **Mesh check:** after the six additional flip cases are accepted, `flip` `k = ±4` on a 1 mm mesh (`-res` halved; 7200 s limit; Slurm `-t 03:00:00`).
   - Every 2 mm vertex is also a 1 mm vertex, so 1 mm heights are restricted to the 2 mm vertices after full-mesh alignment, with mean removed.
   - Project them onto the 2 mm `flip` PCA mean and components.
   - Report quadratic-fit labels, `K_fit`, local sign fractions, PC1 scores and whether `k = −4` still scores below `k = +4`.
   - This does not prove convergence.

Slurm resources: normal partition, one 8-CPU/16 GB task per solve, BLIS single-threaded per project policy.

## Analysis (pre-specified)
- **Height field:** for each accepted shape, rigidly align full-precision deformed state vertices to the flat reference vertices with the calibration's unweighted Kabsch `rigid_fit` on the uniform grid. Height `y(u,v)` is the normal component, with mean removed.
- **Quadratic fit:** over the treated rectangle `|u| ≤ 100`, `|v| ≤ 52` mm, fit `y = c0 + c1 u + c2 v + a u² + b uv + c v²`.
  - Principal curvatures are eigenvalues of `[[2a,b],[b,2c]]`; `K_fit = 4ac − b²`.
  - **Descriptive label:** dome-like if `K_fit > 0`, saddle-like if `K_fit < 0`, with `|κ_min|/|κ_max| ≥ 0.2`. Otherwise label it cylinder-like.
- **Local curvature:** use the reviewed `dome_bo/dome_curvature.geometric_curvature` (15 mm local fits). Report area fractions with positive or negative Gaussian curvature in the treated rectangle.
- **PCA:** if all nine `flip` height fields are accepted, use those nine fields. Do not claim a complete oneface or joint-variant PCA; the incomplete oneface pilot may be reported separately as diagnostics. Report:
  - explained variance;
  - PC1/PC2 maps;
  - scores versus `k`;
  - correlation of PC1 score with `K_fit`.

  With 9 samples, PCA is descriptive.
- **Figures:** shape row, curvature-sign maps, PCA modes, scores versus `k`, plus disclosure of the residual-curl diagnostic and all extrapolations.
