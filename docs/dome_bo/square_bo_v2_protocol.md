# Square-sheet BO v2 — protocol (decisions recorded; not frozen; no launch until implemented and Astra-reviewed)

Status 2026-10-23. Researcher decisions recorded below (answers to the decision questions, this session). Frozen only after the
implementation passes an Astra pre-execution review (`astra-code-reviewer`, `openai-codex/gpt-6-astra`) clears the implementation. Authority
for everything not changed here: "Square-sheet BO stage" in `dome_toolpath_investigation.md` (physics, 9-coordinate
parameterization in `dome_bo/square_hatch.py`, wave of four 8-CPU solves, no-retry rules).
The spent allocation of goal `mvipfud5-fm48ba` does not carry over; this is a new stage.

## Why a new stage

`square_mesh_study_result.md`: the connected-fraction gate is a threshold-and-connectivity statistic that flips on tiny
shape changes (attempt 30: 0.90 / 0.44 / 0.67 at 4 / 2 / 1 mm with the 2 → 1 mm shape RMS 0.002 mm). The 15 mm
local-quadratic estimator also resolves the stroke-scale curvature stripes. Researcher direction (this session): use a
global smooth surface fit as the metric, reward amplitude with a capped sag, and drop the connectivity gate as a
constraint. `dome_bo/surface_fit_scan.py` and `dome_bo/sag_objective.py` show that the degree-8 fit J and the capped
sag are mesh-stable at 2 vs 1 mm (J ≤ 0.003, sag ≤ 0.006 mm, Spearman 1.0 over six candidates).

## Scientific question and outcomes

Can the two-family hatch parameterization on the calibrated bilayer produce a large-amplitude, balanced
(same-sign, similar-magnitude principal curvature) dome under a mesh-stable global metric? Outcomes:
(i) a candidate whose objective is reproduced at 1 mm (verified); (ii) best candidate not reproduced
(diagnose; do not tune on verification); (iii) objective plateaus at a low value (negative result; not proof of impossibility).

## Objective (maximize)

`J_amp = J_fit × min(sag / A_ref, 1)`. A_ref = 1.5 mm (researcher decision 2026-10-23).
- `J_fit`: `dome_continuous_objective.evaluate_dome_objective` (unchanged, frozen file) applied to the principal
  curvatures of a degree-8 total-degree Legendre fit of (x,y,z) in material coordinates (`surface_fit_scan.fit_curvature`),
  central 160 × 160 mm ROI.
- `sag`: |mean fitted height in r < 0.2 h − mean in 0.85 h < r ≤ h| after removing the best plane over the ROI
  (`sag_objective.evaluate`, h = 80 mm).
- No connectivity gate. Reported diagnostics only: connected and total qualifying fractions, saddle fraction, whole-sheet
  maps, fit RMS residual, fold/flip check (`whole_sheet_diagnostics`), unexaggerated shape view.
- Fit RMS is recorded for every attempt but is not a flag or a filter (researcher decision 2026-10-22: no poor-fit
  treatment); the top candidates' maps and unexaggerated shapes are inspected at verification.
- Failed/unaccepted solves have no objective value and are not fitted. There is no feasibility classifier and no
  constraint (researcher decision 2026-10-22: plain qLogEI); all 32 screening and 18 mesh-study solves were accepted.

## Numerics

Mesh 2 mm (`-res .01`), the adaptive-cutback solver through `shellsim.solver_settings.solver_options(tolerance='1e-14')`
(`shellsim/configs/default_adaptive_cutback.json`; one configuration for all drivers, no per-run solver options), absolute gradient gate 1e-14 (the mesh-study 2 mm setting, 170–790 s per solve there). Binary
`run/dome_square/frozen/shell` (hash recorded). Researcher decision 2026-10-23: 2 mm / 1e-14, no 4 mm screening stage.

## Design and policy

- Warm start: the six mesh-study 2 mm cells (attempts 30, 31, 21, 32, 27, 19) already used this physics, mesh, tolerance
  and binary; their retained states are re-scored with `J_amp` (no new solves) and enter the GP as observations 1-6.
  The old 4 mm attempts are not imported (different mesh; attempt 30 changed orientation between 4 and 2 mm).
- Then two fixed anchors as before (p = 8 mm, S = 192, Lv = 192, r_A = r_B = 3, e = 1 and e = 0.4), then
  N_DESIGN = 16 maximin Sobol points (seed 20261024), then qLogEI batches of four (greedy sequential, seeds 20261025+wave; GP on the 9 coordinates, warped as in v1 `square_hatch.warp`).
- Wave = four simultaneous 8-CPU/32 GB srun steps in one 32 CPU/128 GB job; 3900 s per solve step plus 600 s analysis (4500 s combined worker ceiling). CPU solver, so
  GPU scheduling does not apply.
- Infrastructure/analysis error or changed identity: stop, no automatic retry. Attempts reserved on disk before launch.

## Stopping rules (replace the gate-based stop)

Stop after the wave in which the best `J_amp` (warm start included) is at most 5 % above the best before the last 4 BO waves (cumulative over the window, not per wave; design waves do not count); after two
consecutive all-failed waves; at MAX_ATTEMPTS = 96 new solves (the six warm-start cells do not count); on an infrastructure/analysis
error or identity change. Plateau window, 5 % and the 96-solve budget are researcher decisions of 2026-10-23; no allowance carries over from the previous goal.

## Verification (frozen before outcomes)

Re-solve the top 3 distinct recipes by 2 mm `J_amp` at `-res .005` (1 mm, tolerance 1e-14; cells took 2200-4000 s on 8 CPUs).
A candidate is verified if `J_amp` at 1 mm is within 10 % of 2 mm, sag differs by ≤ 0.1 mm (researcher decision 2026-10-23) (fit RMS reported, not a criterion), and
the unexaggerated shape and maps show no fold, edge artifact or isolated bump. Do not retune on verification results.
A claim is "mesh-stable large-amplitude dome under this metric", not an experimental-accuracy claim; physics caveats
of the earlier stage (8 mm width transfer, `center_peak` extrapolation, free-boundary final_only) still apply.

## Implementation to review (no solver code change; implemented 2026-10-23, review pending)

1. `dome_bo/sag_objective.py` (`score()`: J_amp, sag, fit RMS, fractions), `square_v2_attempt.py` (runner: reuses
   `square_attempt.run_attempt`, replaces only the scoring; 2 mm, 1e-14, shared solver config) and `square_v2_campaign.py`
   (`Campaign(square_campaign.Campaign)`: 2 anchors + 16 Sobol design, greedy sequential qLogEI over a 4000-point decoded
   pool with a BoTorch `SingleTaskGP` on standardized warped realized coordinates, plateau stop, 96-attempt cap, root
   `run/dome_square_v2`). The frozen v1 files are unchanged. The fold/flip check and unexaggerated-shape view are not in the
   loop; they are verification-time inspections (`whole_sheet_diagnostics`).
2. Warm start: `warm_start_records()` re-scores the six retained 2 mm cells into `run/dome_square_v2/warm_start.json` and
   refuses cells whose status differs in mesh, tolerance, binary, recipe or state hash. Its sha256 is part of `identity.json`.
3. Tests (`dome_bo/tests/test_square_v2.py`): analytic dome/cylinder/saddle/flat J_amp fixtures, cap behaviour, normal
   reversal, qLogEI batch, controller wiring, plateau rule, nonfinite J_amp as an error, warm-start identity, retained-cell
   regression against `sag_objective.json` (2 mm reproduced, 2 vs 1 mm within 0.01 J_amp and 0.1 mm sag) and a real-solver
   fixture on the adaptive-cutback route. No-retry, error-vs-failure and relative-path behaviour are inherited from
   `square_campaign.Campaign` and covered by `test_square_pipeline.py`.
4. `dome_bo/run_square_v2_campaign.sbatch` (create `run/dome_square_v2/logs` first). `identity.json` (hashes of the
   files above, binary, mesh, tolerance, seeds, warm start) is written at the first launch and checked between waves.

## Decisions

Settled by the researcher (2026-10-22/23): A_ref 1.5 mm; 2 mm / 1e-14 numerics; 96 new solves; plateau rule < 5 % over 4 waves; verification tolerances 10 % / 0.1 mm; plain qLogEI; no poor-fit treatment; adaptive-cutback solver. Astra pre-execution review (2026-10-23): APPROVED_WITH_CAVEATS, no launch blockers, scope limited to this search; evidence `scratch/v2_review_evidence/` (38 tests passed on a compute node; identity hashes). Accepted non-blocking caveats, no code change: warm-start import checks selected fields only (the six cells were compared by hand); `shellsim/solver_settings.py` is imported by `run_dome_continuation.py` but not hashed (unused by `verify()`); the fit RMS is per Cartesian component and sag is fitted world-z height after plane removal (keep these definitions in reports). Open: researcher go-ahead to launch. Current evidence only supports the metric choice on 6 candidates and the 32 screening states.
