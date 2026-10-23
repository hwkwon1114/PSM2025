# Sequence-order pre-execution review

## Verdict: APPROVED_WITH_CAVEATS

**No unresolved scientific or operational launch blockers identified for the frozen 18-case design.** Approval covers the reviewed comparison and declared safeguards only; it does not authorize execution, retries, additional cases, or changes to the existing dome campaign.

## Scope and authority

Reviewed:
- Authoritative goal: `.pi/goals/active_goal_2026101910270493_mvfeqwh7-k4y0nd.md`.
- **Current section only** of `docs/reports/sequence_ablation_v2_spec.md`; historical v2 settings are not execution authority.
- Frozen schedules, recipes, options and identities: `run/sequence_order/checks/frozen_design.json`.
- Runner, batch wrapper, analysis, tests, connected accumulation/solver paths and retained fixtures.
- `docs/reports/dome_linear_repair_review.md`, solely as evidence concerning unchanged numerical machinery—not permission for this investigation.

The scope is twelve primary cases plus six preselected `ab_block` sensitivity cases. No C++ modifications, continuation, extra equilibria, recipe search, or campaign restart are included.

### Recorded identities

SHA256 values below were inspected in the frozen packet and, where available, corroborating execution/repair records. **They were not independently recomputed.**

| Artifact | SHA256 |
|---|---|
| `calibration/configs/dome/baseline_attempt209.json` | `c3211d0b5481237054e891ae69529279d2cf37cdb5f1313f781ff1c5880b1862` |
| `scripts/run_sequence_order.py` | `45a835a91ef21b3610570bfaaddde839687d77a6db92d02e74cc5529eb59d9a0` |
| `scripts/run_sequence_order.sbatch` | `aecbafdf15123fa540a2b83c99068b5b47bf80fd7e88852756f2feadf8d842f1` |
| `calibration/analysis/analyze_sequence_order.py` | `61baa41f3d945ac6c44a078b39aa89eadab473cd5844e843f9898503e33b5f2d` |
| `calibration/tests/test_sequence_order.py` | `93870e00fbd1b08b17d14e12768d9364d35144f7591ffdfbc9f1c80017ff6caa` |
| `src/simulations/Sim_Bilayer_Growth.cpp` | `b7d331d05e614b1ddc44d90f5decd7f755d91434f4202e233f381ab09feb6352` |
| `src/libshell/NativeHybridShellSolver.hpp` | `d71f60390640edf3c1e5335d3cf481e042dfba6eabed5165a2c8d3ecbf6aaf4e` |
| `src/libshell/ReferenceHybrid/Core.hpp` | `f3265448c4d2d210659fab2483d1bf9881852128d1435d7f490e5bc4ae301ebd` |
| `run/native_shape_refinement/build_4375711/bin/shell` | `72ef7d36a1714972bbb5628be9126685f104e13491140f2b419d3f3374d8a27a` |

The packet additionally binds the inspected footprint, hashing, alignment and curvature helpers. Related native headers were inspected for semantics; this was not a complete rebuild-provenance audit. Existing checkout changes outside this slice were not approved or modified.

## Scientific assessment

The design answers the **narrow model-conditional sequence/cadence question**:

- Within each cadence pair, ordered applications and numerical settings are matched; only the timing of equilibrium optimization changes.
- A-only/B-only controls establish actual deformation differences. Perpendicular placement is not evidence that the controls produce contrasting shapes.
- A and B differ in location, dimensions, profile and strip count as well as orientation. This is not an orientation-only causal comparison or fresh validation of transferred calibration parameters.
- Their declared envelopes are disjoint. With material-coordinate coverage, face-local hardening and zero slope feedback, constitutive order effects are not expected when hit sets are disjoint. Confirm the actual hit-set intersection from control outputs. Global elastic coupling and sequential branch selection remain possible.
- Separate lumped runs for all orders are useful implementation/reproducibility checks; no speculative deduplication occurs.
- Sensitivity selection precedes outcomes. The single repeat is a limited repeatability check, not a statistical uncertainty bound; mesh vertices are not independent physical replicates.

Interpretation must distinguish:
1. **Changed cadence-pair targets/history:** confounding or implementation failure, not optimization-path evidence.
2. **Equal objectives and numerically resolved shape differences:** case-specific equilibrium-path/cadence dependence, not physical temporal memory or global optimality.
3. **Equal objectives and common flat starts but differing lumped endpoints:** first investigate numerical reproducibility; ordered loading has supplied no distinct solver starting state.
4. **Failed acceptance or unstable sensitivity:** unresolved, unscored evidence—not a bad-shape result.
5. **Small differences:** agreement at the tested resolution, not exact or universal equivalence.

## Numerical and implementation checks

### Accumulation and cadence

`src/simulations/Sim_Bilayer_Growth.cpp:2383–2841` preserves ordered material hits, both accumulated target metrics and face-shared hit counts. Warm starts retain the complete current vertex/director state; each equilibrium constructs a fresh optimizer. Final-only mode skips intermediate equilibria without resetting accumulated loading.

With `sequence_adaptive=false`, the solve proceeds directly from load fraction zero to one. Exact final metrics replace interpolation before acceptance; failure throws rather than introducing continuation. Seed escape, prescribed deformation and physical release branches are inactive under the reviewed inputs.

The metric update in `src/libshell/GrowthHelper.hpp:708–831` correctly uses
\[
G=R\operatorname{diag}(1+g_1,1+g_2)R^T,\qquad
T=D_m^{-1}GD_m,\qquad
\bar a_{\mathrm{new}}=T^T\bar a_{\mathrm{old}}T.
\]
Positive stretches preserve SPD mathematically; degeneracy and resulting eigenvalues are checked. Native hardening uses the previous face hit count, then increments history for plastic hits. Lumping is therefore not a strain multiplier.

The rectangle initialization is free-boundary. Reference curvature remains fixed; numerical rigid-mode removal is not a persistent physical clamp.

### Acceptance and solver

`scripts/run_sequence_order.py:74–118` checks:
- completion, cycle count and exact equilibrium cadence;
- accepted native gradient-target termination and positive pivots;
- proof-stream agreement, excluding extra/missing native solves;
- exactly one accepted full-load convergence row per planned stage;
- application history and retained intermediate endpoint files.

`src/libshell/NativeHybridShellSolver.hpp:342–361` independently reevaluates the selected endpoint and Hessian. Failed/capped optimization cannot become accepted merely because a retained state looks satisfactory.

The unchanged linear repair uses the correct residual-correction sign, retains only improved finite residuals, and preserves bounded shifts and descent tests. Production arithmetic remains double precision. Positive pivots provide numerical local-stability evidence—not global-minimum certification or geometric-curvature evidence.

### Comparisons

`calibration/analysis/analyze_sequence_order.py:26–108` compares reference coordinates, material coordinates, topology, reference curvature, material constants, both full-precision target metrics and hit counts. Energy differences are withheld when targets/history differ. Target metrics are checked for finite, symmetric SPD values.

Alignment is reference-area-weighted, determinant-positive Kabsch with fixed material correspondence, without scaling, reflection or remapping. Endpoint displacements use native JSON coordinates.

The curvature helper correctly solves the generalized fundamental-form eigenproblem in SI units. Its fixed 15 mm neighborhood, support/rank checks and validity masks must remain visible. Director-based VTP curvature and geometry-only fitted curvature are different estimators.

## Evidence actually examined

- `run/sequence_order/checks/fixture_v1.log`: final invocation reports **six tests passed**; the earlier invocation skipped the retained-fixture test.
- Both folders under `run/sequence_order/fixtures_v1/`: execution/proof/history/convergence evidence shows **four versus one accepted solves**, direct `0→1` loading and zero retries.
- Both histories record six hit events per application and maximum accumulated counts `2,4,6,8`.
- Sequential endpoint gradient: `5.334103715884685e-17`; lumped endpoint gradient: `3.0403980126592196e-15`, both below `5e-14`.
- The passing retained-fixture test asserts exact final objective/history equality.
- `run/native_shape_refinement/build_4375711/native_fixtures_v5.log`: **29 retained native tests passed**.
- Analytic curvature test implementation and prior review evidence were inspected/reused with their existing limitations.

These coarse fixtures establish wiring and bounded correctness, not production acceptance, mesh robustness, or scientific effect size. The complete 18-case `analyze()` workflow was not exercised by this reviewer.

**No commands, builds, simulations, tests, installations, hash computations or scheduler operations were executed.**

## Launch-blocking findings

**None for this reviewed slice.**

## Nonblocking caveats and execution limitations

1. **Identity enforcement is partly procedural.**  
   `scripts/run_sequence_order.py:136–150` records identities and checks binary/input stability during execution, but does not compare every artifact against the frozen packet. Parent preflight should confirm the recorded identities. This approval does not establish continuous immutability or linked-library identity.

2. **Campaign ordering and concurrency are parent responsibilities.**  
   The batch wrapper runs one isolated case; it does not enforce controls-first dispatch, the two-case concurrency maximum, or campaign-wide stopping. Preserve these declared controls and Slurm4467951. No new dispatcher is required.

3. **Timeouts are total-case safeguards.**  
   `scripts/run_sequence_order.py:134` gives external caps of 15,120 seconds for twenty stages, 7,620 for ten, and 870 for one. Native 600-second limits are cooperative, not hard per-stage timers. These fit the declared allocations. The batch file defaults to 4h30m; use the protocol’s 2h30m/30m submission overrides for controls/single-stage cases.

4. **Collection is not the final scientific interpretation.**  
   `calibration/analysis/analyze_sequence_order.py:94–166` retains curvature fields, weights and contrast arrays, but does not produce all promised figures, curvature-distribution summaries or the numerical-resolution decision. Complete those from retained outputs. For `ab_block`, retain the frozen repeatability, panel-diagonal, tighter-tolerance and finer-mesh criteria; do not generalize that check to every ordering.

5. **Initial-state and restart limits remain explicit.**  
   Initial-VTP hash equality alone is not full-precision proof of identical solver DOFs; here it is supported by inspected common initialization and unchanged final-only geometry. General-sequence diagnostics are not restart checkpoints. Interrupted cases remain incomplete and require consultation, not automatic replay.

Approval is limited to the frozen recipes, options, matrix and safeguards. Consequential changes require reassessment. No cumulative historical compute quota is imposed.