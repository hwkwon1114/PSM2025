# Two-gripper paired-release pre-execution review

## Verdict: `APPROVED_WITH_CAVEATS`

**No unresolved launch blockers remain for the reviewed slice.** The first Run 6/Run 9 full-resolution forming trajectories and their four releases are technically ready for parent-authorized launch using the reviewed runner and protocol. No additional fixture or source fix is required before these initial launches.

This verdict does not authorize budget extensions, automatic numerical restarts, changed recipes, scan scoring, or scientific completion.

## Scope, authority and identities

Read the complete objective:

- `.pi/goals/active_goal_2026101623411509_mvbwsnko-2nl477.md`
- Current investigation/Active protocol in `docs/reports/boundary_conditions_and_springback_synthesis.md`
- Prior review: `docs/reports/two_gripper_release_design_review.md`

Historical synthesis sections 1–6 were not used as scientific evidence or authority.

Reviewed the working-tree diff and relevant new checkpoint, runner, configuration and test files against supplied HEAD `e827565a2a4f56c05398789679621abaa24fb28e`. Identities are recorded in:

`run/two_gripper_comparison/checks/preexecution_identity.json`

Principal SHA-256 identities:

| Artifact | SHA-256 |
|---|---|
| `bin/shell` | `d8c2964734bc8afa9b84a8ab88f3172fc901521f267a20e53bb28f072eb45149` |
| `src/libshell/ZigZagSequenceCheckpoint.hpp` | `479e07608d379e1e136722c3a76f695dc233ff2759fbf6d6fc73c8b68e25febc` |
| `src/simulations/Sim_Bilayer_Growth.cpp` | `fb2d7f327ac9271395f6c2322c3c642addc8a9a2b46ed44b41f10e7f49686313` |
| `scripts/run_two_gripper_comparison.py` | `d039e043faa89eebf3d921c51b748bd0e22c32e567e0576b8d989185f518b0b2` |

The manifest also pins both configurations, tests, batch script and fixture execution records.

**Identity-packet defect found and cleared:** the original manifest contained a Python 3.6 traceback rather than JSON. The parent regenerated only that evidence file and attested exact agreement between independently computed `hashlib` and `sha256sum` hashes. I inspected the repaired manifest and matching fixture-recorded identities. My available tools did not permit independent hash execution; that verification remains parent-attested.

## Prior blockers: disposition

### 1. Complete restart state — resolved for this recipe

`src/libshell/ZigZagSequenceCheckpoint.hpp:74–189` preserves full current `[x|y|z|directors]`, both target metrics, shared face history, cycle/proof and immutable reference geometry/directors, topology, UV, curvature and masks.

`getPositions()` includes directors, confirmed in `src/libshell/DCSConfigurations.hpp:335–352`. This implementation does **not** rely on legacy `Mesh` serialization.

Restore checks identity, acceptance, dimensions, finite values, metric positivity, integer history and fixed gripper coordinates before applying decoded state. Unsupported edge/normal clamps are rejected. Publication uses exclusive temporary creation, checksum, fsync and no-overwrite linking.

`src/simulations/Sim_Bilayer_Growth.cpp:2329–2440,2796–2816` restores before growth, skips completed repeats and checkpoints only after accepted full-load targets/history are committed. Cycle-zero reference fields and material hit mapping are reconstructed under identical initialization. This is adequate for the explicitly restricted one-toolpath, warm-start, every-repeat recipe—not a general optimizer restart facility.

### 2. Independent branches — resolved

`src/simulations/Sim_Bilayer_Growth.cpp:2979–3059` restores the immutable clamped checkpoint separately before each release and checks exact recapture equality. Putong’s material-selected A:xyz/B:yz/C:z mask freezes **current release coordinates**; the free branch removes all vertex clamps. Fresh solves prevent optimizer-state contamination.

Fixed coordinates, targets, history and reference curvature are checked unchanged. Neither release adds growth.

### 3. Endpoint acceptance and evidence integrity — resolved

Branches have distinct proof, state, VTP and JSON paths. Failed returned endpoints cannot publish a successful pair. The runner independently checks both endpoint records before marking completion.

The supplied `5e-14` target is not relaxed. Native acceptance requires successful gradient termination, independent residual evaluation and positive reduced-Hessian pivots; capped optimization cannot become accepted merely through endpoint verification.

The curvature check is local numerical evidence, not a global-minimum certificate.

### 4. Provenance and execution bounds — resolved with interpretation below

Inspected saved and derived sequences: only BC fields change; Run 6/9 retain 29/33 repeats and frozen growth/hardening parameters. The runner explicitly selects every-repeat equilibrium and the new configurations rather than replaying Run 9’s archived path alias.

The batch requests normal partition, 8 CPUs, 32 GB and six hours per case. Fresh directories, preserved predecessors and same-identity resume avoid overwriting evidence. There is no automatic process retry.

## Mathematics and diagnostic checks

The rotational constraint matrix implemented in `gaugeGeometry` is consistent with infinitesimal rigid rotation after fixing A. Its determinant is proportional to

\[
b_x(b_xc_y-b_yc_x).
\]

Thus material-space noncollinearity alone does not guarantee deformed gauge rank. Start/end singular-value ratios and actual surviving rigid-mode counts are exported; the native solver adds numerical coordinate conditions only for surviving modes.

Reaction diagnostics genuinely remove vertex masks, refresh caches, evaluate the unchanged state and restore masks with exception handling. Energy/state invariance is checked. The finite-difference fixture tests constrained-coordinate derivatives independently.

The inspected coarse endpoint records show:

- Putong/free remaining modes: **0/6**, at both start and end.
- Identical parent and start-state checksums.
- Accepted gradients approximately **1.98e-17**.
- Nonzero deformed fixed heights, unchanged targets/history and finite unmasked gradients.

These are correctness observations, not production shape-equivalence evidence.

## Evidence and limitations

Inspected:

- `run/two_gripper_comparison/checks/fixtures_final.log`: **32/32** C++ tests passed, including five new checkpoint/diagnostic tests.
- `run/two_gripper_comparison/fixtures/final_{uninterrupted,interrupted,resumed}/` execution records and selected logs/results.
- Python test assertions for frozen inputs, resume equality and branch integrity; **3/3 passing is supplied evidence**, not independently rerun here.

The resumed log begins with cycle 2, consistent with skipping accepted cycle 1. Exact uninterrupted/resumed state/target/history equality is covered by the supplied Python check. The failure-publication path was statically inspected; no dedicated forced-failed-release integration result was supplied.

No builds, simulations, tests, installations or edits were performed by this reviewer. No production runtime feasibility, scan registration, rendered figures or final quantitative analysis was validated.

## Nonblocking caveats and reporting corrections

1. **Timeout wording:** `docs/reports/boundary_conditions_and_springback_synthesis.md:52,68` overstates per-call enforcement. `src/libshell/NativeHybridShellSolver.hpp:349–367` uses cooperative timing: verification may start within `600+30` seconds, and acceptance is rejected after `600+50`; individual evaluations are not preempted. The six-hour Slurm allocation is the outer stop. Continuation retries remain bounded as recorded. Completion within six hours is not demonstrated.

2. **Resume is conditional, not unrestricted permission.** Retain predecessor directories and assess aggregate work before resubmission. Only accepted-repeat restart is exercised; interrupted optimizer state is not resumed.

3. **Correct stale serializer descriptions:** synthesis lines 50/58 incorrectly imply absent director/target serialization. Diagnostic `selected_dofs` includes directors; legacy Mesh writes target forms but overwrites reference/current geometry and lacks the required history/restart contract.

4. **VTP terminology:** `numerical_gauge_active=false` for the free branch means no persistent Putong coordinate mask—not absence of numerical rigid-mode treatment. Use JSON `remaining_rigid_modes_*` and branch identity when reporting.

5. **Rank and reactions require interpretation.** Near-singular gauge geometry can make rank threshold-sensitive. Significant six-coordinate reactions require investigation of residuals, conditioning and symmetry before attributing shape distortion. Label energies/gradients in the normalized **E=1** system.

## Scientific claim boundary

This is a **model-conditional comparison for the researcher-specified geometry**, not historical reproduction or physical identification of Attempt 209 parameters.

Agreement supports equivalence only for these tested endpoints. Disagreement must be distinguished among finite gauge incompatibility, incomplete convergence and different stable equilibria. Rigid-aligned full-field analysis, inspected figures and independent final review remain required. Scan agreement requires separately verified registration and scoring provenance.